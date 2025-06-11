%% Finite Element Modelling
clc
clear
% load stiffness and mass matrix
load M
load K
% node information (column 1-3: node number, lateral coordinate, vertical coordinate)
node(1,1:3)=[1,-3,0]; % beam node
beamL = [0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2];
for i = 2:29
    node(i,1) = i;
    node(i,2) = node(i-1,2)+beamL(i-1);
    node(i,3) = 0;
end
node(30:42,1:3) = [30:42;-1.8*ones(1,13);-0.2:0.1:1.0]';% left tower node
node(43:55,1:3) = [43:55;+1.8*ones(1,13);-0.2:0.1:1.0]';% right tower node
% element information
element(1:28,:) = [1:28;2:29]'; % beam
element(29:40,:) = [30:41;31:42]'; % left tower
element(41:52,:) = [43:54;44:55]'; % right tower
element(53:60,:) = [1,40;4,38;38,10;40,13;17,53;20,51;51,26;53,29]; % cable (from left to right)
% modal analysis
orderN = 10;  % number of adopted modes
[phis,omega] = eigs(K,M,orderN,'sm');
[omega,sort_index] = sort(diag(omega));
freq = sqrt(omega)/2/pi;
phi = zeros(length(K(:,1)),orderN);  % mode shape vector
for i = 1:orderN
    phi(:,i) = phis(:,sort_index(i));
    phi(:,i) = phi(:,i)/max(abs([phi(1:3:end,i);phi(2:3:end,i)]))*0.5; % max normalization based on X or Y mode shape (ignore rotation)
end
Constrained_Dof = [2,11,20,68,77,86,88,89,90,127,128,129]; % delete constrained DOFs
for k=1:1:length(Constrained_Dof)
    phi = insert(phi,Constrained_Dof(k));
end

% clearvars -except M K C freq phi Constrained_Dof node element
%% plot figures
clc
close all

fontsize = 12;
% FEM mode shape
figure
for i=1:9
    subplot(3,3,i)
    axis equal
    %     phi(:,i)=phi(:,i)/norm/2;  % max normalization
    patch('Faces',element,...
        'Vertices',[node(:,2)+[phi(1:3:93,i);phi(95:3:end,i)],node(:,3)+[phi(2:3:93,i);phi(94:3:end,i)]],...
        'FaceColor','white',...
        'EdgeColor','red', ...
        'LineWidth',1);
    hold on
    patch('Faces',element,...
        'Vertices',[node(:,2),node(:,3)],...
        'FaceColor','white',...
        'EdgeColor','k',...
        'LineStyle',':');
    set(gca,...
        'FontName', 'Times New Roman', ...
        'FontSize', fontsize,...
        'Xlim',[-3.5,+3.5],...
        'Ylim',[-0.6,1.2],...
        'Box','On')
    xlabel( 'X(m)', 'FontName', 'Times New Roman', 'FontSize', fontsize);
    ylabel( 'Y(m)', 'FontName', 'Times New Roman', 'FontSize', fontsize);
    title(['Mode ',num2str(i),', Frequency=',num2str(freq(i), '%.4f'),'Hz'],'FontName','Times New Roman', 'FontSize', fontsize);
    % legend('Modeshape','Location','N')

    % set tight layout
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    ax.Position = [outerpos(1) + ti(1), ...
        outerpos(2) + ti(2), ...
        outerpos(3) - ti(1) - ti(3), ...
        outerpos(4) - ti(2) - ti(4)];

end
set(gcf, ...
    'unit','centimeter', ...
    'position',[5,5,28,12])