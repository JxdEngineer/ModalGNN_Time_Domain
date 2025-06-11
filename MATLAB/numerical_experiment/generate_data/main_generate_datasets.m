clc
clear
close all
datasetN = 200;
frequency_out = cell(datasetN,1);
damping_out = cell(datasetN,1);
modeshape_out = cell(datasetN,1);
node_out = cell(datasetN,1);
element_out = cell(datasetN,1);
acceleration_time_out = cell(datasetN,1);

nfft = 1024*2^0;
window = hamming(nfft/4);
tic
MyPar=parpool('local',8); % OPEN PARALLEL CALCULATION
parfor NNN = 1:datasetN
    disp(NNN)
    %%% Geometry model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x1=-20*(1+rand(1));y1=0*(1+rand(1));
    x2=+20*(1+rand(1));y2=0*(1+rand(1));
    x3=+20*(1+rand(1));y3=3*(1+rand(1));
    x4=-20*(1+rand(1));y4=3*(1+rand(1));

    model = createpde;
    rect1 = [3,4,x1,x2,x3,x4,y1,y2,y3,y4]';
    sf = 'rect1';
    ns = char('rect1')';
    g = decsg(rect1,sf,ns);  % read the documentation about 'decsg' function
    pg = geometryFromEdges(model,g);
    mesh = generateMesh(model,"GeometricOrder","linear","Hmax",3);
    node = mesh.Nodes';
    element3 = mesh.Elements';
    % pdeplot(model)

    % Find the unique pair connections
    element2 = [];
    for i = 1:size(element3, 1)
        element2 = [element2; sort(nchoosek(element3(i, :),2),2)];
    end
    element2 = unique(element2,"rows");

    %%% FEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    E = 200*10^9;
    A = 1;
    Rou = 8015.111293*2;    % density of steel
    L = zeros(length(element2),1);
    theta = zeros(length(element2),1);
    for i = 1:length(element2)
        dXY = node(element2(i,1),:)-node(element2(i,2),:);
        L(i) = norm(dXY);
        theta(i) = atan2(dXY(2),dXY(1))/pi*180+180;
    end

    N_DOF = length(node)*3;
    % truss elements
    K_element = cell(length(element2),1);
    T_element = cell(length(element2),1);
    M_element = cell(length(element2),1);
    K_element_transformed = cell(length(element2),1);
    for i = 1:length(element2)
        K_element{i} = Ke_Cable(E*(0.5+rand(1)),A,L(i));
        T_element{i} = Te(theta(i));
        M_element{i} = Me(Rou*A,L(i),0);
    end
    % assemble global matrix
    K = zeros(N_DOF);
    M = zeros(N_DOF);
    for i = 1:length(element2)
        K_element_transformed{i} = T_element{i}'*K_element{i}*T_element{i};
        K = assemble(K,K_element_transformed{i},element2(i,1),element2(i,2));
        M = assemble(M,M_element{i},element2(i,1),element2(i,2));
    end
    K(3:3:end,:) = [];  % delete rotation dof
    K(:,3:3:end) = [];
    M(3:3:end,:) = [];  % delete rotation dof
    M(:,3:3:end) = [];
    % apply constrains
    K_con = K;
    M_con = M;
    % simply supported
    Constrained_DOF = [SearchNode(x1-0.1,x1+0.1,y1-0.1,y1+0.1,node)*2-1,...
        SearchNode(x1-0.1,x1+0.1,y1-0.1,y1+0.1,node)*2,...
        SearchNode(x2-0.1,x2+0.1,y2-0.1,y2+0.1,node)*2];
    % cantilever
    %     Constrained_DOF = [SearchNode(x1-0.1,x1+0.1,y1-0.1,y1+0.1,node)*2-1,...
    %         SearchNode(x1-0.1,x1+0.1,y1-0.1,y1+0.1,node)*2,...
    %         SearchNode(x4-0.1,x4+0.1,y4-0.1,y4+0.1,node)*2-1,...
    %         SearchNode(x4-0.1,x4+0.1,y4-0.1,y4+0.1,node)*2];
    % apply constraints by deleting rows and columns corresponding to constrained dofs
    K_con(Constrained_DOF,:) = []; % delete constrained dof
    K_con(:,Constrained_DOF) = []; % delete constrained dof
    M_con(Constrained_DOF,:) = []; % delete constrained dof
    M_con(:,Constrained_DOF) = []; % delete constrained dof
    % modal analysis
    orders = length(K_con);
    [phis,lamda] = eigs(K_con,M_con,orders,'sm');
    [lamda,sort_index] = sort(diag(lamda));
    freq = sqrt(lamda)/2/pi;
    phi = zeros(length(K_con(:,1)),orders);
    for i = 1:orders
        phi(:,i) = phis(:,sort_index(i))/max(abs(phis(:,sort_index(i))));
    end
    %     clear phis
    for k=1:1:length(Constrained_DOF)
        phi=insert(phi,Constrained_DOF(k));
    end
    % damping matrix
    omega1 = freq(1)*2*pi;omega2 = freq(2)*2*pi;
    ksi1 = 0.01;ksi2 = 0.01;     % damping ratio
    [a0,a1] = DampRayleigh(omega1,omega2,ksi1,ksi2);
    C_con = a0*M_con+a1*K_con;
    ksi_n = a0./(2*freq*2*pi)+a1.*freq*2*pi/2;

    %%% Dynamic analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    deltaT = 1/200;
    T = 20;
    time = 0:deltaT:T;

    P_DOF = SearchNode(x1-0.1,x2+0.1,-0.1,0.1,node)*2;  % excitation on all bottom nodes
    % P_DOF = P_DOF(1:round(length(P_DOF)/2)); % excitation on half bottom nodes
    P_DOF = P_DOF([round(length(P_DOF)/3),round(length(P_DOF)/2)]); % excitation only on mid bottom nodes

    % dynamic loading - free vibration
    % P = zeros(length(node)*2,1);
    % P(P_DOF) = -100*10^5; % Fy
    % P_con = P;
    % P_con(Constrained_DOF,:) = []; % delete constrained dof
    % u0 = K_con^-1*P_con; % calculate static deformation
    % P_con = zeros(length(K_con),length(time));
    % Z0 = zeros(length(K_con),3);
    % Z0(:,1) = u0;

    % dynamic loading - white noise excitations
    P = zeros(length(node)*2,length(time));
    P(P_DOF,:) = wgn(length(P_DOF),length(time),50);
    P_con = P;
    P_con(Constrained_DOF,:) = []; % delete constrained dof
    P_con(:,round(length(P_con)/2):end) = 0;  % simulate unstationary vibration
    Z0 = zeros(length(K_con),3);

    Z = NewmarkConstantAverageAccleration(K_con,M_con,C_con,P_con,Z0,deltaT,T);
    u_con = Z{1};u = u_con;
    v_con = Z{2};v = v_con;
    a_con = Z{3};a = a_con;
    for k = 1:length(Constrained_DOF) % restore constrained dof
        u = insert(u,Constrained_DOF(k));
        v = insert(v,Constrained_DOF(k));
        a = insert(a,Constrained_DOF(k));
    end

    a_y = a(2:2:end,:); % only save acceleration in Y direction
    %     a_y = awgn(a_y,10,'measured');  % add 10% gaussian white noises

    phi_y = phi(2:2:end,:);

    % observe PSD
    % [psd1,f] = pwelch(a_y(10,:),window,[],nfft,1/deltaT);
    % [psd2,f] = pwelch(a_y(20,:),window,[],nfft,1/deltaT);
    % [psd3,f] = pwelch(a_y(30,:),window,[],nfft,1/deltaT);
    % figure
    % hold on
    % plot(f,psd1)
    % plot(f,psd2)
    % plot(f,psd3)
    % plot(f,db(psd1))
    % plot(f,db(psd2))
    % plot(f,db(psd3))

    node_out{NNN} = node;
    element_out{NNN} = element2;
    frequency_out{NNN} = freq;
    damping_out{NNN} = ksi_n;
    acceleration_time_out{NNN} = a_y;
    for k = 1:length(phi(1,:))
        modeshape_out{NNN}(:,k) = phi_y(:,k)/max(abs(phi_y(:,k)))/sign(phi_y(10,k));  % maximum normalization and unify the sign
    end
end
delete(MyPar) %
save dataset node_out frequency_out modeshape_out acceleration_time_out element_out damping_out % output dataset
toc