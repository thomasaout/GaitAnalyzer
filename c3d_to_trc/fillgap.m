function result = fillgap(MATRICE)
% ---------------------------------
% FILLGAP: restaure les données manquantes des trajectoires fournies par
% Vicon, elles se traduisent toujours par des zéros
% - l'algorithme interpole entre tous les zéros des données fournies
% - pour faire face aux zéros aux bords, une procédure d'augmentation des données
%   (identique à celle pour utilisée pour le filtrage) est utilisée


% INPUT:
%   MATRICE 	: temps x Nbre_cannaux
% OUTPUT
%   result 		: matrice de même dimension que MATRICE mais 'néttoyée'
% ----------------------------------
%%

% initialisations
result  = MATRICE;
[nl,nc] = size(MATRICE);

% opération pour chaque colonne
for i = 1:nc
    
    if sum(MATRICE(:,i) == 0) == 0
        result(:,i) = MATRICE(:,i);
        
    elseif sum(MATRICE(:,i) == 0) == nl
        %fprintf('FILLGAP: attention: nullité de la colonne %u\n',num2str(i));
        % do nothing
    else
        % vérifier la non existence de zeros aux bords
        % et utiliser la duplication des données si vrai
        if MATRICE(1,i)==0 || MATRICE(end,i)==0
            try
                [temp,N] = parTrois(MATRICE(:,i));
                temp = interpoler(parTrois(temp));
                result(:,i) = temp(N:end-N);
            catch
               %disp(e.message);
               %fprintf('FILLGAP: impossible de récupérer la colonne %u\n',i);
               result(:,i) = ones(nl,1)*mean(MATRICE(MATRICE(:,i)~=0,i));
            end
            
        else
            result(:,i) = interpoler(MATRICE(:,i));
            
        end
    end
end

%% SOUS FONCTIONS
    function [y,nbre_pts] = parTrois(X,opt)
        % méthode d'Amarantini pour éviter les hystérèses aux bornes
        % INPUT:
        %  X est le vecteur à étendre
        %  opt = est le pourcentage de données à utiliser, par défaut = 0.2
        % OUTPUT
        % y : le vecteur étendu
        % nbre_pts : les nombre de points supplémentaire aux deux bords

        % initialisation
        if nargin <2
            opt =0.2;
        end
        X = X(:);
        N_ = length(X);
        nbre_pts = floor(opt*N_);
       
        % concaténation
        origine1 = mean(X(1:floor(opt*N_/5)));
        origine2 = mean(X(end-floor(opt*N_/5):end));
       
        y = [-flipud(X(1:nbre_pts))+2*origine1;X(2:end-1);-flipud(X(end-nbre_pts:end))+2*origine2]';
    end
    
    % fonction d'interpolation par cubic sline
    function result = interpoler(X)
        x_ref_  = 1:length(X);
        indice  = find(X ~= 0);
        result  = interp1(x_ref_(indice)',X(indice),x_ref_,'splin');
    end



end % end of all