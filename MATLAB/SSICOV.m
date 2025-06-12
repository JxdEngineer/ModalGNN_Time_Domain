function [fn,zeta,phi,varargout] = SSICOV(y,dt,varargin)
%
% -------------------------------------------------------------------------
% [fn,zeta,phi,varargout] = SSICOV(y,dt,varargin) identifies the modal
% parameters of the M-DOF system whose response histories are located in
% the matrix y, sampled with a time step dt.
% -------------------------------------------------------------------------
% Input:
% y: time series of ambient vibrations: matrix of size [MxN]
% dt : scalar: Time step
% Varargin: contains additional optaional parameters:
%	'Ts': scalar : time lag for covariance calculation
%	'methodCOV': scalar: method for COV estimate ( 1 or 2)
%	'Nmin': scalar: minimal number of model order
%	'Nmax': scalar: maximal number of model order
%	'eps_freq': scalar: frequency accuracy
%	'eps_zeta': scalar: % damping accuracy
%	'eps_MAC': scalar: % MAC accuracy
%	'eps_cluster': scalar: % maximal distance inside each cluster
%   'nfft': signal length of FFT operation
% -------------------------------------------------------------------------
% Output:
% fn: eigen frequencies identified
% zeta:  modal damping ratio identified
% phi:mode shape identified
% varargout: structure data useful for stabilization diagram
% -------------------------------------------------------------------------
%  Syntax:
% [fn,zeta,phi] = SSICOV(y,dt,'Ts',30) specifies that the time lag
% has to be 30 seconds.
%
% [fn,zeta,phi] = SSICOV(y,dt,'Ts',30,'Nmin',5,'Nmax',40) specifies that the
% time lag has to be 30 seconds, with a system order ranging from 5 to 40.
%
% [fn,zeta,phi] = SSICOV(y,dt,'eps_cluster',0.05) specifies that the
% max distance inside each cluster is 0.05 hz.
%
% [fn,zeta,phi] = SSICOV(y,dt,'eps_freq',1e-2,'eps_MAC'.1e-2) changes the
% default accuracy for the stability checking procedure
%
% -------------------------------------------------------------------------
% Organization of the function:
% 6 steps:
% 1 - Claculation of cross-correlation function
% 2 - Block hankel assembling and SVD of the block-Hankel matrix
% 3 - Modal identification procedure
% 4 - Stability checking procedure
% 5 - Selection of stable poles only
% 6 - Cluster Algorithm
% -------------------------------------------------------------------------
% References:
% Magalhaes, F., Cunha, A., & Caetano, E. (2009).
% Online automatic identification of the modal parameters of a long span arch
% bridge. Mechanical Systems and Signal Processing, 23(2), 316-329.
%
% Magalhães, F., Cunha, Á., & Caetano, E. (2008).
% Dynamic monitoring of a long span arch bridge. Engineering Structures,
% 30(11), 3034-3044.
% -------------------------------------------------------------------------
% Author: E Cheynet, Universitetet i Stavanger
% Last modified: 03/03/2019
% -------------------------------------------------------------------------
%
% see also plotStabDiag.m

%%
% options: default values
p = inputParser();
p.CaseSensitive = false;
p.addOptional('Ts',500*dt);
p.addOptional('methodCOV',1);
p.addOptional('Nmin',2);
p.addOptional('Nmax',30);
p.addOptional('eps_freq',1e-2);
p.addOptional('eps_zeta',4e-2);
p.addOptional('eps_MAC',5e-3);
p.addOptional('eps_cluster',0.2);
p.addOptional('nfft',1024);
p.parse(varargin{:});

% Number of outputs must be >=3 and <=4.
nargoutchk(3,4)
% size of the input y
[Nyy,N]= size(y);

% shorthen the variables name
eps_freq = p.Results.eps_freq ;
eps_zeta = p.Results.eps_zeta ;
eps_MAC = p.Results.eps_MAC ;
eps_cluster = p.Results.eps_cluster ;
Nmin = p.Results.Nmin ;
Nmax = p.Results.Nmax ;
nfft = p.Results.nfft ;

%  Natural Excitation Technique (NeXT)
[IRF,~] = NExT(y,dt,p.Results.Ts,p.Results.methodCOV);
% Block Hankel computations
[U,S,V] = blockHankel(IRF);
if isnan(U)
    fn = nan;
    zeta = nan;
    phi = nan;
    if nargout==4
        varargout = {nan};
    end
    return
end
% Stability check
kk=1;
for ii=Nmax:-1:Nmin % decreasing order of poles
    if kk==1
        [fn0,zeta0,phi0] = modalID(U,S,V,ii,Nyy,dt);
    else
        [fn1,zeta1,phi1] = modalID(U,S,V,ii,Nyy,dt);
        [a,b,c,d,e] = stabilityCheck(fn0,zeta0,phi0,fn1,zeta1,phi1);
        fn2{kk-1}=a;
        zeta2{kk-1}=b;
        phi2{kk-1}=c;
        MAC{kk-1}=d;
        stablity_status{kk-1}=e;
        fn0=fn1;
        zeta0=zeta1;
        phi0=phi1;
    end
    kk=kk+1;
end

% sort for increasing order of poles
stablity_status=fliplr(stablity_status);
fn2=fliplr(fn2);
zeta2=fliplr(zeta2);
phi2=fliplr(phi2);
MAC=fliplr(MAC);

% get only stable poles
[fnS,zetaS,phiS,MACS] = getStablePoles(fn2,zeta2,phi2,MAC,stablity_status);

if isempty(fnS)
    warning('No stable poles found');
    fn = nan;
    zeta = nan;
    phi = nan;
    if nargout==4
        varargout = {nan};
    end
    return
end

% Hierarchical cluster
[fn3,zeta3,phi3] = myClusterFun(fnS,zetaS,phiS);
if isnumeric(fn3)
    warning('Hierarchical cluster failed to find any cluster');
    fn = nan;
    zeta = nan;
    phi = nan;
    if nargout==4
        varargout = {nan};
    end
    return
end
% save('clusterData.mat','fn3','zeta3')
% average the clusters to get the frequency and mode shapes
for ii=1:numel(fn3)
    fn(ii)=nanmean(fn3{ii});
    zeta(ii)=nanmean(zeta3{ii});
    phi(ii,:)=nanmean(phi3{ii},2);
end

% sort the eigen frequencies
[fn,indSort]=sort(fn);
zeta = zeta(indSort);
phi = phi(indSort,:);

% varargout for stabilization diagram
if nargout==4
    paraPlot.status=stablity_status;
    paraPlot.Nmin = Nmin;
    paraPlot.Nmax = Nmax;
    paraPlot.fn = fn2;
    varargout = {paraPlot};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [U,S,V] = blockHankel(h)
        %
        % [H1,U,S,V] = SSICOV(h) calculate the shifted block hankel matrix H1 and
        % the  result from the SVD of the block hankel amtrix H0
        %
        % Input:
        % h: 3D-matrix
        %
        % Outputs
        % H1: Shifted block hankel matrix
        % U : result from SVD of H0
        % S : result from SVD of H0
        % V : result from SVD of H0
        %%
        if or(size(h,1)~=size(h,2),ndims(h)<3)
            error('the IRF must be a 3D matrix with dimensions <M x M x N> ')
        end
        % get block Toeplitz matrix
        N1 = round(size(h,3)/2)-1;
        M = size(h,2);
        clear H0
        for oo=1:N1
            for ll=1:N1
                T1((oo-1)*M+1:oo*M,(ll-1)*M+1:ll*M) = h(:,:,N1+oo-ll+1);
            end
        end
        if or(any(isinf(T1(:))),any(isnan(T1(:))))
            warning('Input to SVD must not contain NaN or Inf. ')
            U=nan;
            S=nan;
            V=nan;
            return
        else
            try
                [U,S,V] = svd(T1);
            catch exception
                warning(' SVD of the block-Toeplitz did not converge ');
                U=nan;
                S=nan;
                V=nan;
                return
            end
        end
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [IRF,t] = NExT(x,dt,Ts,method)
        %
        % [IRF] = NExT(y,ys,T,dt) implements the Natural Excitation Technique to
        % retrieve the Impulse Response Function (IRF) from the cross-correlation
        % of the measured output y.
        %
        % [IRF] = NExT(y,dt,Ts,1) calculate the IRF with cross-correlation
        % calculated by using the inverse fast fourier transform of the
        % cross-spectral power densities  (method = 1).
        %
        % [IRF] = NExT(y,dt,Ts,2) calculate the IRF with cross-correlation
        % calculated by using the unbiased cross-covariance function (method = 2)
        %
        %
        % y: time series of ambient vibrations: vector of size [1xN]
        % dt : Time step
        % method: 1 or 2 for the computation of cross-correlation functions
        % T: Duration of subsegments (T<dt*(numel(y)-1))
        % IRF: impusle response function
        % t: time vector asociated to IRF
        %%
        if nargin<4, method = 2; end % the fastest method is the default method
        if ~ismatrix(x), error('Error: x must be a vector or a matrix'),end
        [Nxx,N1]=size(x);
        if Nxx>N1
            x=x';
            [Nxx,N1]=size(x);
        end
        
        % get the maximal segment length fixed by T
        M = round(Ts/dt);
        switch method
            case 1
                clear IRF
                for oo=1:Nxx
                    for jj=1:Nxx
                        y1 = fft(x(oo,:));
                        y2 = fft(x(jj,:));
                        h0 = ifft(y1.*conj(y2));
                        IRF(oo,jj,:) = h0(1:M);
                    end
                end
                % get time vector t associated to the IRF
                t = linspace(0,dt.*(size(IRF,3)-1),size(IRF,3));
                if Nxx==1
                    IRF = squeeze(IRF)'; % if Nxx=1
                end
            case 2
                IRF = zeros(Nxx,Nxx,M+1);
                for oo=1:Nxx
                    for jj=1:Nxx
                        [dummy,lag]=xcov(x(oo,:),x(jj,:),M,'unbiased');
                        IRF(oo,jj,:) = dummy(end-round(numel(dummy)/2)+1:end);
                    end
                end
                if Nxx==1
                    IRF = squeeze(IRF)'; % if Nxx=1
                end
                % get time vector t associated to the IRF
                t = dt.*lag(end-round(numel(lag)/2)+1:end);
        end
        % normalize the IRF
        if Nxx==1
            IRF = IRF./IRF(1);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [fn,zeta,phi] = modalID(U,S,V,Nmodes,Nyy,dt)
        %
        % [fn,zeta,phi] = modalID(H1,U,S,V,N,M) identify the modal propeties of the
        % system, given the shifted block hankel matrix H1, and the outputs of the
        % SVD of the vlock hankel matrix H0
        %----------------------------------
        % Input:
        % U: matrix obtained from [U,S,V]=svd(H0) is [N1 x N1]
        % S: matrix obtained from [U,S,V]=svd(H0) is [N1 x N1]
        % V: matrix obtained from [U,S,V]=svd(H0) is [N1 x N1]
        % N: Number of modes (or poles)
        % M: Number of DOF (or sensors)
        %----------------------------------
        % Outputs
        % H1: Shifted block hankel matrix
        % U : result from SVD of H0
        % S : result from SVD of H0
        % V : result from SVD of H0
        % dt: time step
        %----------------------------------
        
        
        if Nmodes>=size(S,1)
            warning(['Nmodes is larger than the numer of row of S. I have to take Nmodes = ',num2str(size(S,1))]);
            % extended observability matrix
            O = U*sqrt(S);
            % extended controllability matrix
            GAMMA = sqrt(S)*V';
        else
            O = U(:,1:Nmodes)*sqrt(S(1:Nmodes,1:Nmodes));
            % extended controllability matrix
            GAMMA = sqrt(S(1:Nmodes,1:Nmodes))*V(:,1:Nmodes)';
        end
        % Get A and its eigen decomposition
        
        IndO = min(Nyy,size(O,1));
        C = O(1:IndO,:);
        jb = round(size(O,1)./IndO);
        A = pinv(O(1:IndO*(jb-1),:))*O(end-IndO*(jb-1)+1:end,:);
        [Vi,Di] = eig(A);
        
        mu = log(diag(Di))./dt; % poles
        fn = abs(mu(2:2:end))./(2*pi);% eigen-frequencies
        zeta = -real(mu(2:2:end))./abs(mu(2:2:end)); % modal amping ratio
        phi = real(C(1:IndO,:)*Vi); % mode shapes
        phi = phi(:,2:2:end);
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [fn,zeta,phi,MAC,stablity_status] = stabilityCheck(fn0,zeta0,phi0,fn1,zeta1,phi1)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % [fn,zeta,phi,MAC,stablity_status] = stabilityCheck(fn0,zeta0,phi0,fn1,zeta1,phi1)
        % calculate the stability status of each mode obtained for
        % two adjacent poles (i,j).
        %
        % Input:
        % fn0: eigen frequencies calculated for pole i: vetor of N-modes [1 x N]
        % zeta0: modal damping ratio for pole i: vetor of N-modes [1 x N]
        % phi0: mode shape for pole i: vetor of N-modes [Nyy x N]
        % fn1: eigen frequencies calculated for pole j: vetor of N-modes [1 x N+1]
        % zeta1: modal damping ratio for pole j: vetor of N-modes [1 x N+1]
        % phi1: mode shape for pole j: vetor of N-modes [Nyy x N+1]
        %
        % Output:
        % fn: eigen frequencies calculated for pole j
        % zeta:  modal damping ratio for pole i
        % phi:mode shape for pole i
        % MAC: Mode Accuracy
        % stablity_status: stabilitystatus
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Preallocation
        stablity_status = [];
        fn = [];
        zeta = [];
        phi = [];
        MAC=[];
        %% frequency stability
        N0 = numel(fn0);
        N1 = numel(fn1);
        for rr=1:N0
            for jj=1:N1
                stab_fn = errCheck(fn0(rr),fn1(jj),eps_freq);
                stab_zeta = errCheck(zeta0(rr),zeta1(jj),eps_zeta);
                [stab_phi,dummyMAC] = getMAC(phi0(:,rr),phi1(:,jj),eps_MAC);
                % get stability status
                if stab_fn==0
                    stabStatus = 0; % new pole
                elseif stab_fn == 1 && stab_phi == 1 && stab_zeta == 1
                    stabStatus = 1; % stable pole
                elseif stab_fn == 1 && stab_zeta ==0 && stab_phi == 1
                    stabStatus = 2; % pole with stable frequency and modalshape vector
                elseif stab_fn == 1 && stab_zeta == 1 && stab_phi ==0
                    stabStatus = 3; % pole with stable frequency and damping
                elseif stab_fn == 1 && stab_zeta ==0 && stab_phi ==0
                    stabStatus = 4; % pole with stable frequency
                else
                    error('Error: stablity_status is undefined')
                end
                fn = [fn,fn1(jj)];
                zeta = [zeta,zeta1(jj)];
                phi = [phi,phi1(:,jj)];
                MAC = [MAC,dummyMAC];
                stablity_status = [stablity_status,stabStatus];
            end
        end
        
        [fn,ind] = sort(fn);
        zeta = zeta(ind);
        phi = phi(:,ind);
        MAC = MAC(ind);
        stablity_status = stablity_status(ind);
        
        function y = errCheck(x0,x1,eps)
            if or(numel(x0)>1,numel(x1)>1)
                error('x0 and x1 must be a scalar');
            end
            if abs(1-x0./x1)<eps % if frequency for mode i+1 is almost unchanged
                y =1;
            else
                y = 0;
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [fnS,zetaS,phiS,MACS] = getStablePoles(fn,zeta,phi,MAC,stablity_status)
        fnS = [];zetaS = [];phiS=[];MACS = [];
        for oo=1:numel(fn)
            for jj=1:numel(stablity_status{oo})
                if stablity_status{oo}(jj)==1
                    fnS = [fnS,fn{oo}(jj)];
                    zetaS = [zetaS,zeta{oo}(jj)];
                    phiS = [phiS,phi{oo}(:,jj)];
                    MACS = [MACS,MAC{oo}(jj)];
                end
            end
        end
        
        % remove negative damping
        fnS(zetaS<=0)=[];
        phiS(:,zetaS<=0)=[];
        MACS(zetaS<=0)=[];
        zetaS(zetaS<=0)=[];
        
        % Normalized mode shape
        for oo=1:size(phiS,2)
            phiS(:,oo)= phiS(:,oo)./max(abs(phiS(:,oo)));
            if diff(phiS(1:2,oo))<0
                phiS(:,oo)=-phiS(:,oo);
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [fn,zeta,phi] = myClusterFun(fn0,zeta0,phi0)
        
        [~,Nsamples] = size(phi0);
        pos = zeros(Nsamples,Nsamples);
        for i1=1:Nsamples
            for i2=1:Nsamples
                [~,MAC0] = getMAC(phi0(:,i1),phi0(:,i2),eps_MAC); % here, eps_MAC is not important.
                pos(i1,i2) = abs((fn0(i1)-fn0(i2))./fn0(i2)) +1-MAC0; % compute MAC number between the selected mode shapes   
            end
            
        end

        if numel(pos)==1
            warning('linkage failed: at least one distance (two observations) are required');
            fn = nan;
            zeta = nan;
            phi = nan;
            return
        else
            Z =  linkage(pos,'single','euclidean');
            myClus = cluster(Z,'Cutoff',eps_cluster,'Criterion','distance');
            Ncluster = max(myClus);
            
            ss=1;
            fn = {};
            for rr=1:Ncluster
                if numel(myClus(myClus==rr))>5
                    dummyZeta = zeta0(myClus==rr);
                    dummyFn = fn0(myClus==rr);
                    dummyPhi = phi0(:,myClus==rr);
                    valMin = max(0,(quantile(dummyZeta,0.25) - abs(quantile(dummyZeta,0.75)-quantile(dummyZeta,0.25))*1.5));
                    valMax =quantile(dummyZeta,0.75) + abs(quantile(dummyZeta,0.75)-quantile(dummyZeta,0.25))*1.5;
                    dummyFn(or(dummyZeta>valMax,dummyZeta<valMin)) = [];
                    dummyPhi(:,or(dummyZeta>valMax,dummyZeta<valMin)) = [];
                    dummyZeta(or(dummyZeta>valMax,dummyZeta<valMin)) = [];
                    fn{ss} = dummyFn;
                    zeta{ss} = dummyZeta;
                    phi{ss} = dummyPhi;
                    ss=ss+1;
                end
            end
            if isempty(fn)
                fn = nan;
                zeta = nan;
                phi = nan;
                return
            end
        end
        
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [y,dummyMAC] = getMAC(x0,x1,eps)
        Num = abs(x0(:)'*x1(:)).^2;
        D1= x0(:)'*x0(:);
        D2= x1(:)'*x1(:);
        dummyMAC = Num/(D1.*D2);
        if dummyMAC >(1-eps)
            y = 1;
        else
            y = 0;
        end
    end
%% Plot stabilization diagram
% close
% fontsize=12;
% % calculate SSA spectrum
% Acc = y';
% window = hanning(nfft/4);
% for I=1:size(Acc,2)
%     for J=1:size(Acc,2)
%         [PSD(I,J,:),F(I,J,:)]=cpsd(Acc(:,I),Acc(:,J),window,[],nfft,1/dt);
%     end
% end
% Frequencies(:,1)=F(1,1,:);
% for I=1:size(PSD,3)
%     [u,s,~] = svd(PSD(:,:,I));
%     s1(I) = s(1);                                                          % First eigen values
%     s2(I) = s(2,2);                                                        % Second eigen values
%     ms(:,I)=u(:,1);                                                        % Mode shape
% end
% s1 = db(s1);
% s1 = s1-min(s1);
% s1 = s1/(max(s1)/Nmax/0.8);
% figure
% hold on
% for k=Nmin:Nmax-1
%     I_f_z_m=(stablity_status{k-Nmin+1}==1);
%     I_f_m=(stablity_status{k-Nmin+1}==2);
%     I_f_z=(stablity_status{k-Nmin+1}==3);
%     I_f=(stablity_status{k-Nmin+1}==4);
%     text(fn2{k-Nmin+1}(I_f),k*ones(1,length(fn2{k-Nmin+1}(I_f))),'f',...
%         'Color','yellow','FontSize',fontsize)
%     text(fn2{k-Nmin+1}(I_f_m),k*ones(1,length(fn2{k-Nmin+1}(I_f_m))),'v',...
%         'Color','blue','FontSize',fontsize)
%     text(fn2{k-Nmin+1}(I_f_z),k*ones(1,length(fn2{k-Nmin+1}(I_f_z))),'d',...
%         'Color','green','FontSize',fontsize)
%     text(fn2{k-Nmin+1}(I_f_z_m),k*ones(1,length(fn2{k-Nmin+1}(I_f_z_m))),'s',...
%         'Color','red','FontSize',fontsize)
% end
% plot(Frequencies(:,1),s1,'LineWidth',1.5,'Color','k')
% 
% % % set(gcf,...
% % %     'Unit', 'Centimeter', ...
% % %     'Position', [20, 5, 7+0, 5+0],...
% % %     'color','w');
% set(gca,...
%     'FontName', 'Times New Roman', ...
%     'FontSize', fontsize, ...
%     'Box', 'On', ...
%     'XGrid','On', ...
%     'YGrid','On', ...
%     'TickDir', 'In', ...
%     'TickLength', [0.01 0.01],...
%     'Ylim',[0,Nmax+5])
% % set(gca,'LineWidth',1)
% % % tag=legend('PSD curve');
% % % set(tag,...
% % %     'FontName', 'Times New Roman', ...
% % %     'FontSize', fontsize,...
% % %     'Location','southeast')
% xlabel( 'Frequency (Hz)', 'FontName', 'Times New Roman', 'FontSize', fontsize);
% ylabel( 'Order', 'FontName', 'Times New Roman', 'FontSize', fontsize);
% % hold off
%% 
end