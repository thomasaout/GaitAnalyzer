function[c3d,erreur,message] = c3dread(nom_fichier_c3d)

erreur = 0;
message='Lecture du fichier c3d réussie';
c3d.prefix_label='';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%c3d.prefix_label
%c3d.nombre_label_analogique
%c3d.taille_nom_canaux
%c3d.nom_analogique(k)
%c3d.nombre_marqueur
%c3d.taille_nom_marqueur
%c3d.nom_marqueur(k)
%c3d.nombre_de_trajectoire
%c3d.nombre_canaux
%c3d.premiere_image
%c3d.derniere_image
%c3d.frequence
%c3d.frequence_analogique
%c3d.temps                              % ici on a des matrices!!!
%c3d.coord(numero_de_trajectoire).data
%c3d.coord(numero_de_trajectoire).databis % je sais pas ce que c'est
%(anciennement les colonnes 4 et 5 de coord.data)
%c3d.anal(canal).data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Ouverture du fichier c3d & intialisation

if nargin ==0
    [f_,pathnamec3d] = uigetfile({'*.c3d','C3d Files (*.c3d)';...
                               '*.*'  ,'Tous les Fichiers (*.*)';},...
                          'C3D READ','fichiers.c3d','MultiSelect', 'off');
    nom_fichier_c3d = char(strcat(pathnamec3d,f_));
end


fichier_c3d=fopen(nom_fichier_c3d,'rb');
if (fichier_c3d ==-1)
    message = 'Impossible d''ouvrir le fichier c3d';
    erreur = 1;
    return;
end

frewind(fichier_c3d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lecture du type de processeur pour la lecture des reels


i = fread(fichier_c3d, 1, 'uint8');
j = fread(fichier_c3d, 1, 'uint8');
if (j==80)
else
    message = 'Le fichier ouvert n''est pas un fichier c3d';
    erreur = 1;
    return;
end

frewind(fichier_c3d);
for k = 1:(i-1)
    fread(fichier_c3d,512,'uint8');
end

fread(fichier_c3d, 1, 'uint8');
fread(fichier_c3d, 1, 'uint8');
fread(fichier_c3d, 1, 'uint8');
processeur = fread(fichier_c3d, 1, 'uint8');

if (processeur == 84)
    formati = 'ieee-le';
    formatr = 'ieee-le';
elseif (processeur == 85)
    formati = 'ieee-le';
    formatr = 'vaxd';
elseif ( processeur == 86)
    formati = 'ieee-be';
    formatr = 'ieee-be';
else
    message = 'Le numéro de processeur ne correspond pas à un type connu';
    erreur = 1;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Lecture du HEADER du fichier C3D

frewind(fichier_c3d);
c3d.header.bloc_parametre = fread(fichier_c3d, 1, 'uint8');
if (c3d.header.bloc_parametre == 1)
    message = 'Fichier c3d sans le bloc header. Lecture impossible';
    erreur = 1;
    return;
end 

c3d.header.c3d = fread(fichier_c3d, 1, 'uint8');
c3d.header.nombre_de_trajectoire = fread(fichier_c3d,1,'uint16',formati);
c3d.nombre_de_trajectoire=c3d.header.nombre_de_trajectoire;

c3d.header.donnees_analogiques_par_image = fread(fichier_c3d,1,'uint16',formati);

c3d.header.premiere_image = fread(fichier_c3d, 1, 'uint16',formati);
c3d.premiere_image = c3d.header.premiere_image;

c3d.header.derniere_image = fread(fichier_c3d, 1, 'uint16',formati);
c3d.derniere_image = c3d.header.derniere_image;
c3d.header.gap = fread(fichier_c3d, 1, 'uint16',formati);

c3d.header.scale = fread(fichier_c3d,1,'float32',formatr);

c3d.header.DATA_START = fread(fichier_c3d, 1, 'uint16',formati);

c3d.header.nombre_de_canaux_analogiques = fread(fichier_c3d, 1, 'uint16',formati);

c3d.header.frequence = fread(fichier_c3d,1,'float32',formatr);
c3d.frequence = c3d.header.frequence;

pos1 = 24;
while (pos1 < 294)
    if ( fread(fichier_c3d, 1, 'uint16',formati)==0);
    else
        message = 'Octet 24 à 294 non nuls.';
        erreur = 1;
        return;
    end
    pos1=pos1+2;
end


c3d.header.label_bloc_present = fread(fichier_c3d, 1, 'uint16',formati);
c3d.header.bloc_label = fread(fichier_c3d, 1, 'uint16',formati);
c3d.header.evenement_present = fread(fichier_c3d,1,'uint16',formati);
c3d.header.nombre_evenement = fread(fichier_c3d,1,'uint16',formati); 
fread(fichier_c3d,1,'uint16',formati);

for i = 1:18
    c3d.header.evenement(i) = fread(fichier_c3d,1,'float32',formatr);
end
for i = 1:18
    c3d.header.evenement_actif(i) = fread(fichier_c3d,1,'uint8');
end
    
fread(fichier_c3d,1,'uint16',formati);
for i = 1:18
    c3d.header.nom_evenement(i)=fread(fichier_c3d,1,'4*uchar=>uchar');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Réinitialisation et passage au début du bloc PARAMETER
frewind(fichier_c3d);
for i = 1:(c3d.header.bloc_parametre-1)
    fread(fichier_c3d,512,'uint8');
end

c3d.parameter.premier_bloc=fread(fichier_c3d, 1, 'uint8');

c3d.parameter.c3d=fread(fichier_c3d, 1, 'uint8');

if ( (c3d.parameter.c3d == 80) || (c3d.parameter.c3d == 0) )
else
    message = 'L''en tête du bloc parametre ne correspond pas à un fichier c3d';
    erreur = 1;
    return;
end

c3d.parameter.nombre_de_bloc_parametre = fread(fichier_c3d, 1, 'uint8');

c3d.parameter.processeur = fread(fichier_c3d, 1, 'uint8');

i = 0;
pointeur = 1;
verif = 0;

while (abs(pointeur) > 0)
    
    taille_nom = fread(fichier_c3d,1,'int8');
    
    ID_NUMBER = fread(fichier_c3d,1,'int8');
    
    if (taille_nom == 0)
        break;
    end
    
    if (ID_NUMBER == 0)
        break;
    end
    
    
    if (ID_NUMBER < 0)      
       
        i = i+1;  
        dim(i) = 1;
        c3d.parameter.groupe(i,1).taille_nom = taille_nom;
        
        c3d.parameter.groupe(i,1).ID_NUMBER = ID_NUMBER;
        
        c3d.parameter.groupe(i,1).nom = fread(fichier_c3d,abs(c3d.parameter.groupe(i,1).taille_nom),'uchar=>uchar');
    
        c3d.parameter.groupe(i,1).groupe_suivant = fread(fichier_c3d,1,'uint16',formati);
        pointeur = c3d.parameter.groupe(i,1).groupe_suivant;
    
        c3d.parameter.groupe(i,1).taille_description = fread(fichier_c3d,1,'uint8');
    
        c3d.parameter.groupe(i,1).description = fread(fichier_c3d,c3d.parameter.groupe(i,1).taille_description,'uchar=>uchar');
        
    elseif (ID_NUMBER > 0)
        
        for boucle = 1:i
            if (abs(c3d.parameter.groupe(boucle,1).ID_NUMBER) == ID_NUMBER )
                dim(boucle) = dim(boucle)+1;
                verif = 1;
                
                c3d.parameter.groupe(boucle,dim(boucle)).taille_nom = taille_nom;
        
                c3d.parameter.groupe(boucle,dim(boucle)).ID_NUMBER = ID_NUMBER;
    
                c3d.parameter.groupe(boucle,dim(boucle)).nom = fread(fichier_c3d,abs(c3d.parameter.groupe(boucle,dim(boucle)).taille_nom),'uchar=>uchar');
    
                c3d.parameter.groupe(boucle,dim(boucle)).groupe_suivant = fread(fichier_c3d,1,'uint16',formati);
                pointeur = c3d.parameter.groupe(boucle,dim(boucle)).groupe_suivant;
    
                c3d.parameter.groupe(boucle,dim(boucle)).data_type = fread(fichier_c3d,1,'int8');
    
                if ( (abs(c3d.parameter.groupe(boucle,dim(boucle)).data_type) < 1) || (c3d.parameter.groupe(boucle,dim(boucle)).data_type > 4) )
                    erreur = 1;
                    return;
                end            
        
                c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension = fread(fichier_c3d,1,'uint8');
        
                if (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension > 0)
                    for k=1:c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension
                        c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(k) = fread(fichier_c3d,1,'uint8');
                        if (c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(k) == 0)
                            message = 'La dimension d''un paramètre vaut 0. Erreur possible';
                        end
                    end
                end
        
                if (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension == 0)
                    if (c3d.parameter.groupe(boucle,dim(boucle)).data_type == -1)
                        c3d.parameter.groupe(boucle,dim(boucle)).data = fread(fichier_c3d,1,'char');
                    elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 1)
                        c3d.parameter.groupe(boucle,dim(boucle)).data = fread(fichier_c3d,1,'int8');
                    elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 2)
                        c3d.parameter.groupe(boucle,dim(boucle)).data = fread(fichier_c3d,1,'int16',formati);
                    elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 4)
                        c3d.parameter.groupe(boucle,dim(boucle)).data = fread(fichier_c3d,1,'float32',formatr);
                    end
                elseif (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension == 1)
                    for k=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(1)
                        if (c3d.parameter.groupe(boucle,dim(boucle)).data_type == -1)
                            c3d.parameter.groupe(boucle,dim(boucle)).data(k) = fread(fichier_c3d,1,'char');
                        elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 1)
                            c3d.parameter.groupe(boucle,dim(boucle)).data(k) = fread(fichier_c3d,1,'int8');
                        elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 2)
                            c3d.parameter.groupe(boucle,dim(boucle)).data(k) = fread(fichier_c3d,1,'int16',formati);
                        elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 4)
                            c3d.parameter.groupe(boucle,dim(boucle)).data(k) = fread(fichier_c3d,1,'float32',formatr);
                        end                         
                    end    
                elseif (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension == 2)
                    for k=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(2)
                        for l=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(1)
                            if (c3d.parameter.groupe(boucle,dim(boucle)).data_type == -1)
                                c3d.parameter.groupe(boucle,dim(boucle)).data(k,l) = fread(fichier_c3d,1,'char');
                            elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 1)
                                c3d.parameter.groupe(boucle,dim(boucle)).data(k,l) = fread(fichier_c3d,1,'int8');
                            elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 2)
                                c3d.parameter.groupe(boucle,dim(boucle)).data(k,l) = fread(fichier_c3d,1,'int16',formati);
                            elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 4)
                                c3d.parameter.groupe(boucle,dim(boucle)).data(k,l) = fread(fichier_c3d,1,'float32',formatr);
                            end                         
                        end
                        if (c3d.parameter.groupe(boucle,dim(boucle)).data_type == -1)
                        end 
                    end
                elseif (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension == 3)
                    for k=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(3)
                        for l=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(2)
                            for m=1:c3d.parameter.groupe(boucle,dim(boucle)).taille_parametre(1)
                                if (c3d.parameter.groupe(boucle,dim(boucle)).data_type == -1)
                                    c3d.parameter.groupe(boucle,dim(boucle)).data(k,l,m) = fread(fichier_c3d,1,'char');
                                elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 1)
                                    c3d.parameter.groupe(boucle,dim(boucle)).data(k,l,m) = fread(fichier_c3d,1,'int8');
                                elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 2)
                                    c3d.parameter.groupe(boucle,dim(boucle)).data(k,l,m) = fread(fichier_c3d,1,'int16',formati);
                                elseif (c3d.parameter.groupe(boucle,dim(boucle)).data_type == 4)
                                    c3d.parameter.groupe(boucle,dim(boucle)).data(k,l,m) = fread(fichier_c3d,1,'float32',formatr);                         
                                end
                            end
                        end
                    end
                elseif (c3d.parameter.groupe(boucle,dim(boucle)).nombre_dimension > 3)
                    message = 'Nombre de dimension d''un paramètre supérieur à 3. Format non pris en charge';
                    erreur = 1;
                    return;
                end
                c3d.parameter.groupe(boucle,dim(boucle)).taille = fread(fichier_c3d,1,'uint8');
        
        
                c3d.parameter.groupe(boucle,dim(boucle)).parametre_description = fread(fichier_c3d,c3d.parameter.groupe(boucle,dim(boucle)).taille,'uchar=>uchar');
              
            end
        end
        if (verif == 0)
            message = 'Paramètre rencontré avant la définition du groupe. Format non reconnu';
            erreur = 1;
            return;
        end
    end
end

c3d.parameter.nombre_groupe = i;
for i=1:c3d.parameter.nombre_groupe
    c3d.parameter.groupe(i,1).nombre_parametre = dim(i);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recherche du début des données dans POINT:DATA_START
verif1 = 0;
frewind(fichier_c3d);
for i =1:c3d.parameter.nombre_groupe
    if (strncmpi(char(c3d.parameter.groupe(i,1).nom),'POINT',5)==1)
        for j = 2:c3d.parameter.groupe(i,1).nombre_parametre
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'DATA_START',10)==1)
                if (c3d.parameter.groupe(i,j).data < 0)
                    c3d.parameter.groupe(i,j).data = c3d.parameter.groupe(i,j).data+65536;
                end                    
                if (c3d.parameter.groupe(i,j).data == c3d.header.DATA_START)
                    verif1 = 1;
                    for k =1:(c3d.header.DATA_START-1)
                        fread(fichier_c3d,512,'uint8');
                    end
                else
                    message = 'POINT:DATA_START <> c3d.header.DATA_START';
                    erreur = 1;
                    return;
                end
            end
        end
    end
end

if (verif1==0)
    erreur = 1;
    return;
end

for i =1:c3d.parameter.nombre_groupe
    if (strncmpi(char(c3d.parameter.groupe(i,1).nom),'SUBJECTS',8)==1)
        for j = 2:c3d.parameter.groupe(i,1).nombre_parametre
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'LABEL_PREFIXES',14)==1)
                c3d.prefix_label= strrep(sprintf('%s',char(c3d.parameter.groupe(i,j).data')),' ','');% enlever le transpose à data et le char()!!!
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verifications
verif1 = 0;
verif2 = 0;
verif3 = 0;
verif4 = 0;
verif5 = 0;
verif6 = 0;
if (c3d.header.nombre_de_canaux_analogiques == 0)
    nombre_anal=0;
else
    nombre_anal=c3d.header.donnees_analogiques_par_image/c3d.header.nombre_de_canaux_analogiques;
end;
freq_anal = c3d.header.nombre_de_canaux_analogiques;


for i =1:c3d.parameter.nombre_groupe
    if (strncmpi(char(c3d.parameter.groupe(i,1).nom),'POINT',5)==1)
        for j = 2:c3d.parameter.groupe(i,1).nombre_parametre
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'SCALE',5)==1)
                if (c3d.parameter.groupe(i,j).data == c3d.header.scale)
                    verif1 = 1;
                end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'USED',4)==1)
                if (c3d.parameter.groupe(i,j).data < 0)
                    c3d.parameter.groupe(i,j).data = c3d.parameter.groupe(i,j).data+65536;
                end
                if (c3d.parameter.groupe(i,j).data == c3d.header.nombre_de_trajectoire)
                        verif2 = 1;
                    end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'RATE',4)==1)
                if (c3d.parameter.groupe(i,j).data == c3d.header.frequence)
                    verif3 = 1;
                end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'FRAMES',6)==1)
                if (c3d.parameter.groupe(i,j).data < 0)
                    c3d.parameter.groupe(i,j).data = c3d.parameter.groupe(i,j).data+65536;
                end
                if (c3d.parameter.groupe(i,j).data == c3d.header.derniere_image-c3d.header.premiere_image+1 )
                    verif4 = 1;
                end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'LABELS',6)==1)
                c3d.nombre_marqueur = c3d.parameter.groupe(i,j).taille_parametre(2);
                c3d.taille_nom_marqueur = c3d.parameter.groupe(i,j).taille_parametre(1);
                for k = 1:c3d.parameter.groupe(i,j).taille_parametre(2)
                    c3d.nom_marqueur(k,1:c3d.taille_nom_marqueur) = char(c3d.parameter.groupe(i,j).data(k,1:c3d.parameter.groupe(i,j).taille_parametre(1)));
                end
            end
        end    
    elseif (strncmpi(char(c3d.parameter.groupe(i,1).nom),'ANALOG',6)==1)
        for j = 2:c3d.parameter.groupe(i,1).nombre_parametre
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'USED',4)==1)
                if (c3d.parameter.groupe(i,j).data < 0)
                    c3d.parameter.groupe(i,j).data = c3d.parameter.groupe(i,j).data+65536;
                end
                if (c3d.header.nombre_de_canaux_analogiques == 0)
                    verif5 = 1;
                else
                    if (c3d.parameter.groupe(i,j).data == c3d.header.donnees_analogiques_par_image/c3d.header.nombre_de_canaux_analogiques)
                        verif5 = 1;
                    end                    
                end                
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'RATE',4)==1)
                if (c3d.parameter.groupe(i,j).data == c3d.header.nombre_de_canaux_analogiques*c3d.header.frequence)
                    verif6 = 1;
                end
                
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'GEN_SCALE',4)==1)
                gen_scale = c3d.parameter.groupe(i,j).data;
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'SCALE',4)==1)
                if (c3d.parameter.groupe(i,j).taille_parametre(1) >= nombre_anal)
                    for anal=1:nombre_anal
                        scale(anal)=c3d.parameter.groupe(i,j).data(anal);
                    end
                else
                    message = 'Nombre de paramètres d''echelle inférieurs au nombre de canaux analogiques';
                    erreur = 1;
                    return;
                end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'OFFSET',4)==1)
                if (c3d.parameter.groupe(i,j).taille_parametre(1) >= nombre_anal)
                    for anal=1:nombre_anal
                        offset(anal)=c3d.parameter.groupe(i,j).data(anal);
                    end
                else
                    message = 'Nombre de paramètres d''offset inférieurs au nombre de canaux analogiques';
                    erreur = 1;
                    return;
                end
            end
            if (strncmpi(char(c3d.parameter.groupe(i,j).nom),'LABELS',6)==1)
                c3d.nombre_label_analogique = c3d.parameter.groupe(i,j).taille_parametre(2);
                c3d.taille_nom_canaux = c3d.parameter.groupe(i,j).taille_parametre(1);
                for k = 1:c3d.parameter.groupe(i,j).taille_parametre(2)
                    c3d.nom_analogique(k,1:c3d.taille_nom_canaux) = char(c3d.parameter.groupe(i,j).data(k,1:c3d.parameter.groupe(i,j).taille_parametre(1)));
                end
            end
        end
    end
end

if (verif1==0)
    message = 'POINT:SCALE <> c3d.header.SCALE';
    erreur = 1;
    return;
end
 
if (verif2==0)
    message = 'POINT:USED <> c3d.header.nombre_de_trajectoire';
    erreur = 1;
    return;
end

if (verif3==0)
    message = 'POINT:RATE <> c3d.header.frequence';
    erreur = 1;
    return;
end

if (verif4==0)
    message = 'POINT:FRAMES <> c3d.header.derniere_image-c3d.header.premiere_image+1';
    erreur = 1;
    return;
end

if (verif5==0)
    message = 'ANALOG:USED <> c3d.header.donnees_analogiques_par_image/c3d.header.nombre_de_canaux_analogiques';
    erreur = 1;
    return;
end

if (verif6==0)
    message = 'ANALOG:RATE <> c3d.header.nombre_de_canaux_analogiques*c3d.header.frequence';
    erreur = 1;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lecture  des données

A = (4*c3d.header.nombre_de_trajectoire+freq_anal*nombre_anal);T = (c3d.header.derniere_image-c3d.header.premiere_image+1);
if (c3d.header.scale < 0)
    data  =  fread(fichier_c3d,A*T,'float32',formatr);
elseif (c3d.header.scale > 0)
    data  =  fread(fichier_c3d,A*T,'int16',formati);
end
data = reshape(data,A,T)';
for j=1:c3d.header.nombre_de_trajectoire
    %c3d.coord(j).data = data(:,4*(j-1)+1:4*j-1);
    c3d.coord(j).data =data(:,1:3);data = data(:,5:end);
end
for anal = 1:nombre_anal
    c3d.analog(anal).data = data(:,(0:(freq_anal-1))*nombre_anal+anal)';
    c3d.analog(anal).data = (c3d.analog(anal).data(:)-offset(anal))*gen_scale;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fermeture et dernières assignations
c3d.nombre_canaux = nombre_anal;
c3d.frequence_analogique = c3d.header.nombre_de_canaux_analogiques*c3d.header.frequence;
fclose('all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












