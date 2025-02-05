function c3d2trc(NB_PTS,prefixe)
% C3D2TRC: CONVERSION D'UN FICHIER C3D EN UN FICHIER LISIBLE PAR OPENSIM (.TRC)
% va chercher les fichiers C3D et les converti en TRC avec le même nom et
% dans le même fichier
% 
% INPUT: 
% NB_PTS est le nombre de point sur lequel on normalise les données dans
% le temps
% prefixe: SI un préfixe existe ! (utilisé pour différentier différents sujets sur un fichier c3d)
% 
% nécessite les fonctions FILLGAP.m et UTIL.m
% ==========================
% First Input is the c3d file and the second input is the osim model.


if nargin <2
    prefixe = '';
        if nargin <1
             NB_PTS = [];
        end
end

u = util();

%% coeffs pour les centres articulaires - ref (Adjustments to McConville et al. and Young et al. body segment inertial parameters, Dumas et Chèze 2007)
% ------------------------
try
    COEFF =  evalin('base','COEFF');
    warning('C3D2TRC: UTILISATION DES COEFFICIENTS DE REGRESSION TROUVES DANS LE WORKSPACE (effacer COEFF du ws et reprendre si erreur)!');
catch
    COEFF.('LHJC') = [ 0.208    -0.361    -0.2780]; % Dumas et Chèze 2007
    COEFF.('RHJC') = [ -0.208   -0.361    -0.2780]; % Dumas et Chèze 2007
    COEFF.('RSJC') = [ 0.0351    0.0421   -0.0006]; % best_regression
    COEFF.('LSJC') = [ 0.7771    0.2104    0.0010]; % best_regression
    COEFF.('CJC')  = [ 0.4511    0.1872   -0.0025]; % best_regression
    %COEFF.('HV')   = [ 0.4928    0.0239    0.0062]; % best_regression(SEL,OCC,RTEMP,HV);
end
 
%% aller chercher les fichiers
% -----------------------------
[f_,pathnamec3d] = uigetfile({'*.c3d','C3d Files (*.c3d)';...
                           '*.*'  ,'Tous les Fichiers (*.*)';},...
                      'CONVERSION C3D -> TRC','fichiers.c3d','MultiSelect', 'on');
f = {};% le ou les fichiers
if iscell(f_)
    for s_= 1:size(f_,2)
        f{s_} =char(strcat(pathnamec3d,f_(s_)));
    end
elseif ischar(f_)
        f{1} = char(strcat(pathnamec3d,f_));
else
    return
end

%% aller chercher le model

% LE CHEMIN DU MODELE EST A INDIQUER ICI 
% C:\TON CHEMIN
% nom = getMarkerFromModel('C:\DATA\matlab_projects\WBM\data\whole_body_model.osim');

% 
[f_,pathname] = uigetfile({'*.osim','OpenSim Model(*.osim)';...
                           '*.*'  ,'Tous les Fichiers (*.*)';},...
                      'CONVERSION C3D -> TRC','Opensim model.osim','MultiSelect', 'off');
nom = getMarkerFromModel(char(strcat(pathname,f_)));

         
%% -------------------------------

for i = 1:length(f)
    
    % charger le fichier c3d
    c3d  = c3dread(f{i});
    % aller chercher les noms des marqueurs dans le c3d
    % et vérifier qu'ils  correspondent à ceux du model
   % idx    = false(1,length(nom));
    NOM    = char(c3d.nom_marqueur);
    liste  = {};
    for k = 1:size(NOM,1)
        temp = NOM(k,:);
        temp = temp(temp~=' ');
        liste{k} = temp;
    end
    
    if nargin < 2
        idx = strfind(c3d.prefix_label,':');
        if length(idx)==1                
            prefixe = liste{1}(1:idx);
        elseif length(idx)>1
            l = 1;
            liste_prefix = {};
            for k = 1:length(idx)
                liste_prefix{k} = c3d.prefix_label(l:idx(k));
                l = idx(k)+1;
            end
            l = listdlg('PromptString','sélectionner un sujet:',...
                'SelectionMode','single',...
                'ListString',liste_prefix);
            prefixe = [liste_prefix{l}];
        end   
    end
    

	%---------------------------
    % écrire dans un fichier TRC
    % --------------------------
    [pathstr,name,~] = fileparts(f{i});
    if isempty(prefixe)
        nom_fichier = [pathstr '/' name '.trc'];
    else
        nom_fichier = [pathstr '/' name '_' prefixe(1:end-1) '.trc'];
    end
    fichier  = fopen(nom_fichier,'wt+');
    if (fichier == -1)
        fclose('all');
        return;
    end
	% -------------------------
    % création du Header du TRC
    %---------------------------
    fprintf(fichier,'PathFileType\t4\t(X/Y/Z)\t%s\n', [name '.trc']);
    fprintf(fichier,'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n');
    if isempty(NB_PTS)
        fprintf(fichier,'%3.2f\t%3.2f\t%u\t%u\tm\t%3.2f\t%u\t%u\n',c3d.frequence,c3d.frequence,c3d.derniere_image,length(nom),c3d.frequence,c3d.premiere_image,c3d.derniere_image); 
    else
        tf = (c3d.derniere_image-1)/c3d.frequence; % modification pour normalisation sur NB_PTS pts
        fprintf(fichier,'%3.2f\t%3.2f\t%u\t%u\tm\t%3.2f\t%u\t%u\n',200/tf,200/tf,200,length(nom),c3d.frequence,c3d.premiere_image,c3d.derniere_image);
    end
    fprintf(fichier,'Frame#\tTime');
    for k = 1:length(nom)
        fprintf(fichier,'\t%s\t\t',nom{k});
    end
    fprintf(fichier,'\n\t');
    
	%---------------------------
    % extraction des données Xi Yi Zi
    % ---------------------------
    M = [];
    for k = 1:length(nom)
        % inscrire Xi Yi et Zi dans le fichier
        fprintf(fichier,'\tX%u\tY%u\tZ%u',k,k,k);
        % extraire les données du c3d pour chaque marqueur
        idx = num(nom{k});
        if ~isempty(idx)
            TEMP = fillgap(c3d.coord(idx).data)/1000;% en mètre + fill gaps
            %TEMP = c3d.coord.data/1000;
        else    % on essaie de reconstruire avec les points disponibles
            try 
                switch nom{k}
                    case 'REJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('RLHE')).data) + 0.5*fillgap(c3d.coord(num('RMHE')).data) )/1000;
                    case 'LEJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('LLHE')).data) + 0.5*fillgap(c3d.coord(num('LMHE')).data) )/1000;
                    case 'RWJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('RRS')).data)  + 0.5*fillgap(c3d.coord(num('RUS')).data)  )/1000;
                    case 'LWJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('LRS')).data)  + 0.5*fillgap(c3d.coord(num('LUS')).data)  )/1000;
                    case 'RAJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('RSPH')).data) + 0.5*fillgap(c3d.coord(num('RLM')).data)  )/1000;
                    case 'LAJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('LSPH')).data) + 0.5*fillgap(c3d.coord(num('LLM')).data)  )/1000;
                    case 'RKJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('RLFE')).data) + 0.5*fillgap(c3d.coord(num('RMFE')).data) )/1000;
                    case 'LKJC'
                        TEMP = ( 0.5*fillgap(c3d.coord(num('LLFE')).data) + 0.5*fillgap(c3d.coord(num('LMFE')).data) )/1000;
                    case 'RSJC'
                        a    = fillgap(c3d.coord(num('RA')).data)/1000;
                        b    = fillgap(c3d.coord(num('LA')).data)/1000;
                        c    = fillgap(c3d.coord(num('SUP')).data)/1000;
                        TEMP = inv_regression(a,b,c,COEFF.('RSJC'));
                    case 'LSJC'
                        a    = fillgap(c3d.coord(num('RA')).data)/1000;
                        b    = fillgap(c3d.coord(num('LA')).data)/1000;
                        c    = fillgap(c3d.coord(num('SUP')).data)/1000;
                        TEMP = inv_regression(a,b,c,COEFF.('LSJC'));
                    case 'RHJC'
                        a    =  (c3d.coord(num('RASIS')).data(:,1)-c3d.coord(num('LASIS')).data(:,1))*COEFF.('RHJC');
                        TEMP= (c3d.coord(num('RASIS')).data + a)/1000;
                    case 'LHJC'
                       a=(c3d.coord(num('RASIS')).data(:,1)-c3d.coord(num('LASIS')).data(:,1))*COEFF.('LHJC');
                       TEMP= (c3d.coord(num('LASIS')).data + a)/1000;
                       
                    case 'RTT2'
                        RTJC = ( 0.5*fillgap(c3d.coord(num('RMFH1')).data) + 0.5*fillgap(c3d.coord(num('RMFH5')).data) )/1000;
                        a =  RTJC - fillgap(c3d.coord(num('RCAL')).data)/1000;
                        TEMP = RTJC  + 0.3*a;
                    case 'LTT2'
                        LTJC = ( 0.5*fillgap(c3d.coord(num('LMFH1')).data) + 0.5*fillgap(c3d.coord(num('LMFH5')).data) )/1000;
                        a =  LTJC - fillgap(c3d.coord(num('LCAL')).data)/1000;
                        TEMP = LTJC  + 0.3*a;
                    case 'RFT3'
                        % RFJC = ( 0.5*fillgap(c3d.coord(num('RHMH2')).data) + 0.5*fillgap(c3d.coord(num('RHMH5')).data) )/1000;
                        RFJC = ( 0.5*fillgap(c3d.coord(num('RHMH1')).data) + 0.5*fillgap(c3d.coord(num('RHMH5')).data) )/1000;
                        RWJC = ( 0.5*fillgap(c3d.coord(num('RRS')).data)  + 0.5*fillgap(c3d.coord(num('RUS')).data)  )/1000;
                        a    = RFJC - RWJC ;
                        TEMP = RFJC  + a;
                    case 'LFT3'
                        %LFJC = ( 0.5*fillgap(c3d.coord(num('LHMH2')).data) + 0.5*fillgap(c3d.coord(num('LHMH5')).data) )/1000;
                        LFJC = ( 0.5*fillgap(c3d.coord(num('LHMH1')).data) + 0.5*fillgap(c3d.coord(num('LHMH5')).data) )/1000;
                        LWJC = ( 0.5*fillgap(c3d.coord(num('LRS')).data)  + 0.5*fillgap(c3d.coord(num('LUS')).data)  )/1000;
                        a    = LFJC - LWJC ;
                        TEMP = LFJC  + a;
                    otherwise
                        warning('marqueur %s non référencé dans c3d2trc (à rajouter pour permettre sa reconstruction)',nom{k});
                       % M = [M zeros(c3d.derniere_image,3)];
                       M = [M zeros(1+c3d.derniere_image,3)];
                end
            catch err
                warning('fcn c3d2trc: problème dans la reconstruction du marqueur %s',nom{k});
                disp(err.message)
                M = [M zeros(c3d.derniere_image,3)]; % modif
            end
        end
        X = TEMP(:,1);
        Y = -TEMP(:,2);
        Z = TEMP(:,3);
        M = [M X Z Y]; % concaténer dans la matrice M et réorienter pour opensim
    end
	
	% réduction éventuelle de la quantité de données
    if ~isempty(NB_PTS)
        M = u.normalisation(M,NB_PTS);
        t = linspace(0,tf,NB_PTS);
        frame = 1:NB_PTS;
    else
        t = ((c3d.premiere_image-1):(c3d.derniere_image-1))/c3d.frequence;
        frame = c3d.premiere_image:c3d.derniere_image;
    end
    % filtrage des données à 6 Hz
    M = u.filtre(4,2*6/c3d.frequence,'low',M);
    
    % concaténation dans une matrice contenant le temps et les "frames"
    try
        M = [frame(:) t(:) M];
    catch err
        fprintf('petit problème! %s\n',err.message);
        M = [frame(:) t(:) u.normalisation(M,length(t))];
    end
	
	% écrire la matrice M (ligne par ligne) dans le fichier
    fprintf(fichier,'\n\n');
    fprintf(fichier,['%u\t%f' repmat('\t%f',1,3*length(nom)) '\n'],M');
    fclose(fichier);
    
    fprintf('Conversion vers opensim terminée pour %s\n',nom_fichier);
 

  
end

% ====================
%% sous fonctions
% ====================

    function n = num(name)
    % numéro du point nommé
        n = find(strcmp([prefixe name],liste),1);
    end

    function nom = getMarkerFromModel(nom_fichier)
    % obtenir les noms des marqueurs présents dans un modèle opensim

        if (nargin == 1)
        else
            error('c3d2tcr::getMarkerFromModel: un seul argument attendu'); 
        end


        nom = {};

        % ====================================
        % ouverture du fichier et initialisation
        % ====================================
        fichier = fopen(nom_fichier,'rb');
        if (fichier == -1)
            error('impossible d''ouvrir le fichier...');
        end
        % charger tout le fichier en entier
        txt = textscan(fichier,'%s','Delimiter','\n');
        fclose(fichier);


        txt = char(txt{:});
        N   = size(txt,1);

        i_ = 1;
        while  ~contains(txt(i_,:),'<MarkerSet>')
            i_ = i_+1;
            if i_>=N
                warning('fcn getMarkerFromModel: AUCUN MARQUEUR DANS LE FORMAT ATTENDU DANS %s\n',nom_fichier)
                return
            end
        end
        k_ = 1;
        txt_ = '';
        while ~contains(txt_,'/MarkerSet') && i_<N
             txt_ = txt(i_,:);
             if contains(txt_,'Marker') && contains(txt_,'name') && contains(txt_,'=')
                 idx_ = strfind(txt_,'"');
                 if length(idx_)== 2
                    nom{k_} = txt_(idx_(1)+1:idx_(2)-1);
                    k_ = k_ + 1;
                 end
             end
             i_ = i_+ 1;
        end
    end
        

end