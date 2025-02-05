function methods = util
%%  UTIL - util
% f = util()  %crée la fonction

% modification: mars 2014

  methods = struct('angle_entre',@angle_entre,...
  'integre',@integre,...
  'trapeze',@trapeze,...
  'vect',@vect,...
  'p_vect',@p_vect,...
  'antisym',@antisym,...
  'scalaire',@scalaire,...
  'normalisation',@normalisation,...
  'filtre',@filtre,...
  'parTrois',@parTrois,...
  'img2eps',@img2eps,...
  'decoupe',@decoupe,...
  'get_file',@get_file,...
  'get_file2',@get_file2,...
  'put_file',@put_file,...
  'clean_matrice',@clean_matrice,...
  'per_cir',@per_cir,...
  'grisage',@grisage,...
  'str2rgb',@str2rgb,...
  'geninv',@geninv,...
  'filtre2',@filtre2,...
  'extremum',@extremum,...
  'pattern_moyen',@pattern_moyen,...
  'filtre_ordre',@filtre_ordre,...
  'npmean',@npmean,...
  'normalize',@normalize,...
  'notchfilter',@notchfilter);
  

%%              calcul de l'angle entre deux veteurs avec atan
    function [y] = angle_entre(v1,v2)
        s = sign(vect(v1,v2));
        y  = 2*atan(s(1)*norm(vect(v1,v2))./(norm(v1)*norm(v2)),scalaire(v1,v2)./(norm(v1)*norm(v2)));
        %y = angle(s(1)*norm(vect(v1,v2))/(norm(v1)*norm(v2)),scalaire(v1,v2)/(norm(v1)*norm(v2)));
    end

    % y compris entre [-pi; pi] si on veut y > pi on prend y = 2pi -
    % asin(sinteta)

%%              intégration: méthode d'Euler
    function [y] = integre(f,dt,fo)
      y(1) = fo;
      for i = 2:size(f,1)
        y(i) = y(i-1) + f(i-1)*dt;
       end
    end
%% integration par méthode des trapèzes améliorés sur des données tabulées
    function INTEGRALE = trapeze(y,dt)
        n = length(y);
        d1 = [-25 48 -36 16 -3]*[y(1) y(2) y(3) y(4) y(5)]'./(12*dt);% les dérivées aux extrémités
        d2 = [ -1 6 -18 10 3]*[y(n-4) y(n-3) y(n-2) y(n-1) y(n)]'./(12*dt);

        INTEGRALE = (y(1)+y(n)+sum(2*y(2:n-1)))*dt/2  - (dt^2/12)*(d2-d1);
    end


%%              Calcul du produit vectoriel AxB 
    function [y] = vect(A,B)
        y = [A(2)*B(3) - A(3)*B(2),A(3)*B(1) - A(1)*B(3),A(1)*B(2) - A(2)*B(1)];
    end
%% opérateur antisymétrique (3D)
    function M = antisym(u)
        %u = [ux uy uz] et u vect qqch = [U]*qqch
        M = [0 -u(3) u(2);
             -u(3) 0 u(1);
             -u(2) u(1) 0];
    end
        

%%              produit vectoriel sur des collonnes
    function [y] = p_vect(U,V)
        for i = 1:size(U,1)
          y(i,:) = vect(U(i,:),V(i,:));
        end
    end

%%              produit scalaire de A.B
    function [y] = scalaire(A,B)
        y = [sum((A.*B)')]';
        %y = A'*B;
    end
%%              fonction de normalisation sur x

% La méthod de fenètrage initialement programmée posait des pbs si length(X)
% n'était pas un multiple de lengh(fenètre)cad qu'on avait à redistribuer le résidus
% On résout ce probème en choisissant des point pivot equidistant autour
% desquels les points seront moyennés (d'où la perte d'information) :
% pt(i) = moyenne( pt(i-pas) : pt(i) +pas).
% voir aussi: resample.mat downsample ou decimate


    function [vecteur_norm] = x_normalisation(X,N,sous_ech)
        X =X(:)';
        if nargin <3
            sous_ech = 0;
        end
        pas = length(X)/N;
%           1)cas ou on veut réduire le nombre de valeur         
        if (length(X)>N)

              if pas < 1
                  pas = 2;  % gestion des exceptions
              end 
              vecteur_index = floor(pas * (1:N));
              
              if sous_ech %cas d'un sous échantillonage
                  vecteur_norm = X(vecteur_index);
              else
                  vecteur_norm = X(vecteur_index(2:end-1));
                  for i  = 1:floor(pas/2)
                    vecteur_norm = vecteur_norm  + X(vecteur_index(2:end-1)+i) + X(vecteur_index(2:end-1)-i);
                  end
                  vecteur_norm = [X(floor(pas));vecteur_norm(:)/(2*floor(pas/2)+1);X(end-floor(pas))];% pas spéciaux pour les extrémités
              end

%           2)cas ou on veut l'augmenter: il s'agit d'une simple
%           interpolation
        else
              x = (0:length(X)-1);%x = (1/pas)*(0:length(X)-1);
              vecteur_norm = interp1(x,X,linspace(0,length(X)-1,N),'splin');% il existe une erreur xi(end)= x(end)-1/N
        end
    end
%% Normalisation de plusieurs colonnne (+général que celui d'en haut...
    function [y] = normalisation(X,N,sous_ech)
        y = [];
        if nargin <3
            sous_ech = 0;
        end
        for i = 1:size(X,2)
            y(:,i) = x_normalisation(X(:,i),N,sous_ech);
        end
    end

%%                  procédure de filtrage (butterworth)

%%           BUTTER de matlab...
    function [filt] = filtre(ordre,fc,type,donnee,opt)
        if nargin <5
            opt = 0.2;% pourcentage en plus pour le par trois
        end
        if size(fc,1)>1
            donnee = filtre(ordre,fc(2:end,:),type,donnee,opt);
        end

        [nl,nc] = size(donnee);
        [b a] = butter(ordre,fc(1,:),type) ;
        for i = 1:nc
            [temp,N] = parTrois(donnee(:,i),opt);
            temp = filtfilt(b,a,temp(:));
            filt(:,i) = temp(N:end-N);
        end
    end
    function [filt] = filtre2(ordre,fc,type,donnee) 
      [nl,nc] = size(donnee);
      [b a] = butter(ordre,fc,type) ;
      for i = 1:nc
        filt(:,i) = filtfilt(b,a,donnee(:,i));
      end
    end
    function filt = notchfilter(fc,width,data)
        % fc et width sont normalisées:
        % si on veu enlever 50Hz  à une fréquence d'échantillonage de 1000
        % fc = 2*50/1000
        % pour le width, pour 3Hz d'amplitude on mettra
        % width = 3/1000
        nc= size(data,2);
        [b,a] = iirnotch(fc,width); 
        for i = 1:nc
             filt(:,i) = filtfilt(b,a,data(:,i));
        end
    end
%%  filtre arbitraire
    function [filt] = filtre_pattern(ordre,pattren,donnee,opt)
        if nargin <5
            opt = 0.2;% pourcentage en plus pour le par trois
        end
      [nl,nc] = size(donnee);
      % a:numérateur de la fonction de transfert
      % b: dénominateurs
      a = 1;
      b = cfirpm(38, [-1 -0.5 -0.4 0.3 0.4 0.8], ...% les pourcentage
               {'multiband', [5 1 2 2 2 1]}, [1 10 5]);
      % fvtool(b,a); % pour voir la réponse du filtre
      for i = 1:nc
        [temp,N] = parTrois(donnee(:,i),opt);
        temp = filtfilt(b,a,temp(:));
        filt(:,i) = temp(N:end-N);
      end
    end

%%          méthode d'Amarantini pour éviter les hystérèses aux bornes
    function [y,nbre_pts] = parTrois(X,opt)
        if nargin <2
            opt = 0.2;
        end
        X= X(:);
        N = length(X);
        nbre_pts = floor(opt*N);
        origine1 = mean(X(1:floor(opt*N/4)));% X(1)
        origine2 = mean(X(end-floor(opt*N/4):end));% X(end)
        y = [-flipud(X(1:nbre_pts))+2*origine1;X(2:end-1);-flipud(X(end-nbre_pts:end))+2*origine2]';
    end
%%               convertion en .eps
% c''est un format d'image utilisé dans les postscript/DVI ...

    function img2eps()
        [filename, pathname] = uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files';...
          '*.*','All Files' },'sélectionne une image...','MultiSelect', 'on');
      if iscellstr(filename)  
          for i =1:max(size(filename))
                str = strcat(pathname,filename(i));
                [path,name,ext] = fileparts(char(str));
                image(imread([path,'\',name,ext]));
                axis off

                saveas(gcf,[path,'\',name,'.eps'],'psc2');
          end
      else
          str = strcat(pathname,filename);
          [path,name,ext] = fileparts(char(str));
          image(imread([path,'\',name,ext]));
          axis off

          saveas(gcf,[path,'\',name,'.eps'],'psc2');
          
      end
        disp('done')

    end

    function [morceau] = decoupe(abscisse,X,sigle,seuil)     
%% morceau(i).data et morceau(i).abscisse
% DONNE LES MORCEAUX DE LA FONCTION QUI SATISFONT AU CRITERE ET LE PLOT:
% le sigle = {'=';'>';'<'}; exemple : morceau.(i).data > seuil
        switch sigle
            case '='
                vect_ind = find(X == seuil);
                id = find(diff(vect_ind)>1);
                morceau(1).data = X(vect_ind(1):vect_ind(id(1)));
                morceau(1).abscisse = abscisse(vect_ind(1):vect_ind(id(1)));
                for i = 1:length(id)-1
                    morceau(i+1).data = X(vect_ind(id(i)+1):vect_ind(id(i+1)));
                    morceau(i+1).abscisse = abscisse(vect_ind(id(i)+1):vect_ind(id(i+1)));
                end
                morceau(length(id)+1).data = X(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
                morceau(length(id)+1).abscisse = abscisse(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
            case '>'
                vect_ind = find(X > seuil);
                id = find(diff(vect_ind)>1);
                morceau(1).data = X(vect_ind(1):vect_ind(id(1)));
                morceau(1).abscisse = abscisse(vect_ind(1):vect_ind(id(1)));
                for i = 1:length(id)-1
                    morceau(i+1).data = X(vect_ind(id(i)+1)-5:vect_ind(id(i+1))+5);
                    morceau(i+1).abscisse = abscisse(vect_ind(id(i)+1)-5:vect_ind(id(i+1))+5);
                end
                morceau(length(id)+1).data = X(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
                morceau(length(id)+1).abscisse = abscisse(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
            case '<'
                vect_ind = find(X < seuil);
                id = find(diff(vect_ind)>1);
                morceau(1).data = X(vect_ind(1):vect_ind(id(1)));
                morceau(1).abscisse = abscisse(vect_ind(1):vect_ind(id(1)));
                for i = 1:length(id)-1
                    morceau(i+1).data = X(vect_ind(id(i)+1):vect_ind(id(i+1)));
                    morceau(i+1).abscisse = abscisse(vect_ind(id(i)+1):vect_ind(id(i+1)));
                end
                morceau(length(id)+1).data = X(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
                morceau(length(id)+1).abscisse = abscisse(vect_ind(id(length(id))):vect_ind(length(vect_ind)));
            case '--'
                k = 0;id = [];
                d = derivation();
                acc = d.deriv(X,1);
                for j = 1:length(X)-1
                    if acc(j+1)*acc(j)<0 && acc(j+1)-acc(j)>0 % on est sur un pic
                        id(k+1)=j+1;
                        k = k+1;
                    end
                end
                if isempty(id) || length(id)<2
                    morceau = [];
                    fprintf(' impossible de découper selon le critère');
                else
                    for i = 1:length(id)-1
                        morceau(i).data = X(id(i):id(i+1)-1);
                        morceau(i).abscisse = abscisse(id(i):id(i+1)-1);
                    end
                end
        end
        
    end
%% fonctions d'accès rapide au path d'une fonction
% attention c'est une structure de données
    function nom_fichier = get_file2(chemin) 
        nom_fichier = {};
        if nargin == 0
            chemin = '';
        end
        [filename, pathname] = uigetfile({'*.*','Tous les Fichiers (*.*)';...
            '*.c3d','C3d Files (*.c3d)';...
            '*.jpg;*.tif;*.png;*.gif','All Image Files';...
            '*.m;*.fig;*.mat;*.mdl','MATLAB Files (*.m,*.fig,*.mat,*.mdl)';}...
                ,'File Selector','multiselect','on',chemin);
            if iscell(filename)
              for i= 1:size(filename,2)
                  nom_fichier{i} =char(strcat(pathname,filename(i)));
              end
            elseif ischar(filename)
                nom_fichier{1} = char(strcat(pathname,filename));
            else
                return
            end
    end
% ici renvoie un string
    function nom_fichier = get_file(chemin) 
        if nargin == 0
            chemin = '';
        end
        [filename, pathname] = uigetfile({'*.*','Tous les Fichiers (*.*)';...
            '*.c3d','C3d Files (*.c3d)';...
            '*.jpg;*.tif;*.png;*.gif','All Image Files';...
            '*.m;*.fig;*.mat;*.mdl','MATLAB Files (*.m,*.fig,*.mat,*.mdl)';}...
                ,'File Selector',chemin);
          nom_fichier = [pathname,filename];
    end
    function  nom_fichier_ecriture = put_file(name)
        [file,path] = uiputfile({'*.*','Tous les Fichiers (*.*)';...
            '*.c3d','C3d Files (*.c3d)';...
            '*.jpg;*.tif;*.png;*.gif','All Image Files';...
            '*.m;*.fig;*.mat;*.mdl','MATLAB Files (*.m,*.fig,*.mat,*.mdl)';},'Enregistrer sous',name);
        nom_fichier_ecriture=[path,file];
    end
%%  clean_matrice (M) interpole entre les valeurs manquantes (NaN et 0)
%   utile dans les dans données Vicon notamment qui remplace les valuers
%   manquantes par des zeros
% si l'option est mise à zéro, les zeros ne sont pas considérés et on
% interpole qu'entre les valeurs manquantes = NaN
    function [y] = clean_matrice(MA_MATRICE,option)
        if nargin <2
            option=1;
        end
        [nl,nc] = size(MA_MATRICE);
        y = MA_MATRICE;
        if option
            MA_MATRICE(MA_MATRICE==0)=NaN;
        end
        lesNaN = isnan(MA_MATRICE(:));
        if sum(lesNaN)>0
            if sum(lesNaN)/length(MA_MATRICE(:))>0.3 % plus de 30 % de données manquantes
                errordlg('Données corrompues : impossible de récupérer les données','Erreur');
                return;
            end
        end
        for i = 1:nc
          if ~any(isnan(MA_MATRICE(:,i)))
            y(:,i) = MA_MATRICE(:,i);
          elseif all(isnan(MA_MATRICE(:,i)))
            disp(['attention collonne ',num2str(i),' enlevée']);
          else
              if isnan(MA_MATRICE(1,i))|| isnan(MA_MATRICE(end,i))
                  DATA = MA_MATRICE(:,i);
                  nbre_pts = floor(0.2*length(DATA));
                  origine1 = DATA(find(isnan(DATA),1,'first'));
                  origine2 = DATA(find(isnan(flipud(DATA)),1,'first'));
                  DATA = [-flipud(DATA(1:nbre_pts))+2*origine1;DATA(2:end-1);-flipud(DATA(end-nbre_pts:end))+2*origine2]';
                  ordre= ~isnan(DATA);
                  x_ref = 1:length(DATA);
                  DATA =interp1(x_ref(ordre)',DATA(ordre),x_ref,'splin');
                  y(:,i) = DATA(nbre_pts:end-nbre_pts);
              else
                  ordre= ~isnan(MA_MATRICE(:,i));
                  x_ref = 1:nl;
                  y(:,i) =interp1(x_ref(ordre)',MA_MATRICE(ordre,i),x_ref,'splin');      
              end
          end
        end
    end
%%  fonction de permutation circulaire
% j'ai pas trouver mieux que d'utiliser une condition
    function y = per_cir(a,b)
        if mod(a,b)==0
            y=b;
        else
            y = mod(a,b);
        end
    end
%% colore l'espace entre deux fonctions données (moyenne + écartype)
    function grisage(x,moyenne,ecart_type,couleur,couleurbis)
        couleurbis = couleur;
        if isempty(couleurbis)
            couleurbis = couleur;
        end
        if ~all(size(couleur)==[1 3])
            try
                couleur = str2rgb(couleur);
                couleurbis = str2rgb(couleurbis);
            catch
                error('couleur non reconnue');
            end
        end
        x =x(:);
        x2 = flipud(x(:));
        m_plus_et = moyenne(:)+ecart_type(:);
        m_moins_et = flipud(moyenne(:)-ecart_type(:));
        hold 'on';
        obj = fill([x',x2'],[m_plus_et' m_moins_et'],couleur,'LineStyle','none');
        plot(x,moyenne,'color',couleurbis,'linewidth',2);
        alpha(obj,0.6);
            set(gcf,'Renderer', 'Painters');
    end
    function col = str2rgb(str)
        switch str
            case 'y'
                col = [1 1 0];
            case 'm'
                col = [1 0 1];
            case 'c'
                col = [0 1 1];
            case 'r'
                col = [1 0 0];
            case 'g'
                col = [0 1 0];
            case 'b'
                col = [0 0 1];
            case 'w'
                col = [1 1 1];
            case 'k'
                col = [0 0 0];
            case 'gris'
                col = [1 1 1]*0.7;
        end
    end
%%                  algorithme de P.Courrieu pour le calcul de la pseudo inverse de Moore-Penrose
    function Y = geninv(G)
        % Returns the Moore-Penrose inverse of the argument
        % Transpose if m < n
        [m,n]=size(G); transpose=false;
        if m<n
            transpose=true;
            A=G*G';
            n=m;
        else
            A=G'*G;
        end
        % Full rank Cholesky factorization of A
        dA=diag(A); tol= min(dA(dA>0))*1e-9;
        L=zeros(size(A));
        r=0;
        for k=1:n
            r=r+1;
            L(k:n,r)=A(k:n,k)-L(k:n,1:(r-1))*L(k,1:(r-1))';
            % Note: for r=1, the substracted vector is zero
            if L(k,r)>tol
                L(k,r)=sqrt(L(k,r));
                if k<n
                    L((k+1):n,r)=L((k+1):n,r)/L(k,r);
                end
            else
                r=r-1;
            end
        end
        L=L(:,1:r);
        % Computation of the generalized inverse of G
        M=inv(L'*L);
        if transpose
            Y=G'*L*M*M*L';
        else
            Y=L*M*M*L'*G';
        end
    end
%%
    function [UP_,DOWN_] = extremum(data,logique)
        %renvoie les indices des extremums d'une fonction
        if nargin<2
            logique = 0;% n'utilise pas l'indiçage logique
        end
         if size(data,2)>1
            UP_  = {};
            DOWN_ = {};
        else
            UP_ = [];
            DOWN_ = [];
        end
        for k = 1:size(data,2)
            datat = data(:,k);
            try 
                d = derivation();
                EXACT = 1;
            catch e
                EXACT = 0;
            end
            if EXACT
                d1  = d.vitesse(datat(:),1);
                d2  = d.acceleration(datat(:),1);
                if logique
                    UP = d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)<0 == 1;
                    DOWN = d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)>0 == 1;
                else
                    UP = find(d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)<0 == 1);
                    DOWN = find(d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)>0 == 1);
                end
            else
                d1  = diff(datat,1);
                d2 = diff(datat,2);
                if logique
                    UP = d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)<0 == 1;
                    DOWN = d1(1:end-1).*d1(2:end)<0 & d2(1:end-1)>0 == 1;
                else
                    UP = find(d1(1:end-1).*d1(2:end)<0 & d2<0 == 1);
                    DOWN = find(d1(1:end-1).*d1(2:end)<0 & d2>0 == 1);
                end
            end
            if size(data,2)>1
                UP_(k)  = {UP(:)};
                DOWN_(k) = {DOWN(:)};
            else
                UP_ = UP;
                DOWN_ = DOWN;
            end
        end
    end
%%
    function [Y,stdY,mm] = pattern_moyen(X,longueur_cycle,option)
        if nargin<3
            option = 0;
        end
        mm = [];
        % renvoie le pattern moyen
        nbre_cycle = floor(size(X,1)/longueur_cycle);
        
        Somme = zeros(longueur_cycle,size(X,2));
        Somme_carre = zeros(longueur_cycle,size(X,2));
        for i = 1:nbre_cycle
            temp = X((i-1)*longueur_cycle+1:i*longueur_cycle,:);
            Somme = Somme + temp;
            Somme_carre = Somme_carre+ temp.^2;
            mm(i,:) = max(temp);
        end
        stdY = sqrt((Somme_carre-(Somme.^2)/nbre_cycle)/(nbre_cycle-1));
        Y = Somme/nbre_cycle;
        if option
            if size(X,1)==longueur_cycle
                Y = X./repmat(max(Y),longueur_cycle,1);
            else
                mm = median(mm);% normalisé par la madiane des pics
                Y = Y./repmat(mm,size(Y,1),1);
                stdY = stdY./repmat(mm,size(Y,1),1);
            end
        end
    end

    % une autre fonction de  des patterns
    function Y = normalize(X,typef,longueur_cycle)
        if any(strcmpi(typef,{'max','mean','median'}))
            eval(['Y  = bsxfun(@rdivide,X,',typef,'(X));']);
        else
            nbre_cycle = size(X,1)/longueur_cycle;
            m = [];
            for i = 1:nbre_cycle
                temp = X((i-1)*longueur_cycle+1:i*longueur_cycle,:);
                m(i,:) = max(temp);
            end
            if strcmp(typef,'mean_')
                Y = bsxfun(@rdivide,X,mean(m));
            elseif strcmp(typef,'median_')
                Y = bsxfun(@rdivide,X,median(m));
            end
        end
    end
            
            

    function Y = filtre_ordre(ordre,X,seuil)
        % filtre la nieme variation et intègre pour renvoyer le version
        % filtrée au ieme ordre
        % (mon premier algorithme récursif au fait!!)
        Y(1,:) = X(1,:);
        d = diff(X);
        if ordre==2
            d = filtre(2,seuil,'low',d);
        elseif ordre>2
            d = filtre_ordre(ordre-1,d,seuil);
        end
        Y = cumsum([Y(1,:);d]);
    end
%%
    function M = npmean(DD)
    % npmean (Non Perturbed Mean)
    % c'est une moyenne robuste qui écarte les outliers, mais qui suppose une
    % valeur fixe corrompue par du bruit
    % auteur : NA Turpin (2014)
        maxiter = 20;
        for  k = 1:size(DD,2)
            data = DD(:,k);
            ii   = isnan(data);
            data = data(~ii);
            % estimation 1
            m_old = mean(data);
            for i =1:maxiter
                % virer les outiers
                poids = (data-m_old)/sqrt(sum((data-m_old).^2)/length(data));% zscore
                data = data(poids<1.96);
                % estimation
                poids = 1./abs(data-m_old);
                m = sum(poids.*data./(sum(poids)));
                % test
                if abs(m-m_old)/m_old<1E-3
                    break;
                end
                % update
                m_old = m;
            end
            M(:,k)=m;
        end
    end  
            

      
end
