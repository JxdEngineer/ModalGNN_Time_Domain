%% modal identification - numerical test
clc
clear
close all

load("C:\Users\xudjian\Desktop\truss_200_lowpass.mat")
fs = 200;

segment_index = 1001:3000;

t = 1/fs*[0:length(acceleration_time_out{1}(1,segment_index))-1];

% test_no = 1:100;
% test_no = 101:200;

% test_no = 1:3; % demonstrate identified phi of truss 3 in the paper
test_no = 1:8; % demonstrate identified phi of truss 3 in the paper

testN = length(test_no);

Phi_id_MAC = zeros(testN,4);
Freq_id = zeros(testN,4);
Zeta_id = zeros(testN,4);

Zeta_true = zeros(testN,4);
Freq_true = zeros(testN,4);

Zeta_id_RE = zeros(testN,4);
Freq_id_RE = zeros(testN,4);

tic
for i = 1:testN %length(damping_out)
    disp(['i=',num2str(i)])

    Phi_true = modeshape_out{test_no(i)}(:,1:4);
    Zeta_true(i,:) = damping_out{test_no(i)}(1:4)*100'; % record zeta in percentage
    Freq_true(i,:) = frequency_out{test_no(i)}(1:4)';

    node_N = length(acceleration_time_out{test_no(i)}(:,1));
    node_mask = ones(node_N,1);
    % evenly remove some feature nodes, only 18% remains %%%%%%%%%%%%%%
    missing_indices = 1:2:node_N;
    node_mask(missing_indices) = 0;
    missing_indices = 1:3:node_N;
    node_mask(missing_indices) = 0;
    missing_indices = 2:3:node_N;
    node_mask(missing_indices) = 0;

    node_mask(acceleration_time_out{test_no(i)}(:,1)==0) = 1; % boundary condition is known

    node_mask = logical(node_mask);
    acceleration_incomplete = acceleration_time_out{test_no(i)}(node_mask,segment_index);

    % Automated EFDD
    % [phi,freq,zeta] = AFDD(acceleration_incomplete,t,10, ... % identify more modes for redundancy
    %     'PickingMethod','auto', ...
    %     'Ts', 10);

    % SSI
    [freq,zeta,phi] = SSICOV(acceleration_incomplete,1/fs, ...
        'Nmin',2,'Nmax',50);

    % match automatically identified frequencies with true frequencies %%%%
    Phi_id_incomplete = zeros(length(acceleration_incomplete(:,1)),4);
    for j = 1:4
        [~,I] = min(abs(Freq_true(i,j)-freq));
        Phi_id_incomplete(:,j) = phi(I,:)';
        Zeta_id(i,j) = zeta(I)*100; % damping ratios in %
        Freq_id(i,j) = freq(I);
    end
    % directly use the first four identified modes %%%%%%%%%%%%%%%%%%%%%%%
    % Phi_id_incomplete = phi(1:4,:)';
    % Zeta_id(i,1:4) = zeta(1:4)*100; % damping ratios in %
    % Freq_id(i,1:4) = freq(1:4);

    % use inter/extrapolation to recover complete mode shapes on every node
    Phi_id = zeros(node_N,4);
    node_incomplete = node_out{test_no(i)}(node_mask,:);
    for j = 1:4
        phi_complete = scatteredInterpolant(node_incomplete(:,1),node_incomplete(:,2),Phi_id_incomplete(:,j),'natural','linear');
        Phi_id(:,j) = phi_complete(node_out{test_no(i)}(:,1),node_out{test_no(i)}(:,2));
    end

    for j = 1:4
        % Phi_id_MAC(i,j) = MAC(real(Phi_id(node_mask,j)),Phi_true(node_mask,j));% use known DOFs
        Phi_id_MAC(i,j) = MAC(real(Phi_id(:,j)),Phi_true(:,j));% use interpolated full-field DOFs
        Zeta_id_RE(i,j) = (Zeta_id(i,j)-Zeta_true(i,j))/Zeta_true(i,j)*100;
        Freq_id_RE(i,j) = (Freq_id(i,j)-Freq_true(i,j))/Freq_true(i,j)*100;
    end
end
toc

% observe PSD
% signal = acceleration_time_out{1}(3, :);
% M = 2^(nextpow2(numel(t)/8));
% [PSD,f] = cpsd(signal,signal,M,round(M/2),M,fs);
% figure
% plot(f,PSD)

% observe mode shapes
figure
for j = 1:4
    subplot(2,2,j)
    hold on
    % patch('Faces',element_out{i},...
    %     'Vertices',[node_out{i}(:,1),node_out{i}(:,2)],...
    %     'Marker','o');
    scatter(node_out{i}(node_mask,1),node_out{i}(node_mask,2)-Phi_id(node_mask,j),'filled','MarkerFaceColor','b')
    patch('Faces',element_out{i},...
        'Vertices',[node_out{i}(:,1),node_out{i}(:,2)-Phi_id(:,j)],...
        'Marker','o',...
        'edgecolor','b');
    patch('Faces',element_out{i},...
        'Vertices',[node_out{i}(:,1),node_out{i}(:,2)+Phi_true(:,j)],...
        'Marker','o',...
        'edgecolor','g');
    axis equal
    legend('identified','true')
    xlim([-40,40])
    ylim([-5,10])
    xlabel('X (m)')
    ylabel('Y (m)')
end
%% modal identification - laboratory test
clc
clear
close all

load('C:\Users\xudjian\Desktop\ModalGNN_Time_Domain\Python\dataset2.mat')
fs = 200;

t = 1/fs*[0:length(acceleration_time_out{1}(1,:))-1];

testN = 10;

Phi_id_MAC = zeros(testN,4);
Freq_id = zeros(testN,4);
Zeta_id = zeros(testN,4);

test_no = 1:testN;

tic
for i = test_no %length(damping_out)
    disp(['i=',num2str(i)])

    Phi_true = modeshape_out{i}(:,1:4);
    % Zeta_true = damping_out{i}(1:4);
    Freq_true = frequency_out{i}(1:4);

    node_N = length(acceleration_time_out{i}(:,1));
    node_mask = zeros(node_N,1);
    node_mask([1,4,5,7,9,12,15,18,21,23,26,29]) = 1;
    node_mask = logical(node_mask);
    acceleration_incomplete = acceleration_time_out{i}(node_mask,:);

    % Automated EFDD
    % [phi,freq,zeta] = AFDD(acceleration_incomplete,t,10, ... % identify more modes for redundancy
    %     'PickingMethod','auto', ...
    %     'Ts', 10);

    % SSI
    [freq,zeta,phi] = SSICOV(acceleration_incomplete,1/fs, ...
        'Nmin',2,'Nmax',50);

    % match automatically identified frequencies with true frequencies
    for j = 1:4
        [~,I] = min(abs(Freq_true(j)-freq));
        Phi_id_incomplete(:,j) = phi(I,:)';
        Zeta_id(i,j) = zeta(I)*100; % damping ratios in %
        Freq_id(i,j) = freq(I);
    end

    % use inter/extrapolation to recover complete mode shapes on every node
    Phi_id = zeros(node_N,4);
    node_incomplete = node_out{i}(node_mask,:);
    for j = 1:4
        Phi_id(:,j) = interp1(node_incomplete(:,1),Phi_id_incomplete(:,j),node_out{i}(:,1),'linear');
    end

    for j = 1:4
        Phi_id_MAC(i,j) = MAC(real(Phi_id(:,j)),Phi_true(:,j)); % use all DOFs
        % Phi_id_MAC(i,j) = MAC(real(Phi_id(node_mask,j)),Phi_true(node_mask,j));% use known DOFs
        % Zeta_RE(i,j) = (Zeta_id(j)-Zeta_true(j))/Zeta_true(j)*100;
        % Freq_RE(j) = (Freq_id(i,j)-Freq_true(j))/Freq_true(j)*100;
    end

end
toc

% visualize identified mode shapes
figure
ttt = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact'); % Adjusts subplot spacing
title(ttt, 'Test 20'); % Set the main title for the entire figure
for j = 1:4
    phi_pred = real(Phi_id(:,j))/max(abs(real(Phi_id(:,j))))/sign(real(Phi_id(9,j)));
    phi_true = Phi_true(:,j)/max(abs(real(Phi_true(:,j))))/sign(real(Phi_true(9,j)));
    ax = nexttile;
    hold on
    plot(node_out{i}(:,1),phi_pred,'-o','Color','b','LineWidth',1.5)
    plot(node_out{i}(:,1),phi_true,'-*','Color','r','LineWidth',1.5)
    scatter(node_out{i}(node_mask,1),phi_pred(node_mask),'filled','MarkerFaceColor','g')
    set(gca, ...
        'xgrid','on', ...
        'ygrid','on')
    box on
    legend('phi_{pred}','phi_{true}','known')
    title(ax, ['MAC=',num2str(Phi_id_MAC(i,j),'%.6f'),', Freq=',num2str(Freq_id(i,j),'%.4f')])
    % set tight layout
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    ax.Position = [outerpos(1) + ti(1), ...
        outerpos(2) + ti(2), ...
        outerpos(3) - ti(1) - ti(3), ...
        outerpos(4) - ti(2) - ti(4)];
end
sgtitle(['Test ',num2str(i)])


% visualize true mode shapes
figure
ttt = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact'); % Adjusts subplot spacing
for j = 1:3
    phi_true = Phi_true(:,j)/max(abs(real(Phi_true(:,j))))/sign(real(Phi_true(9,j)));
    ax = nexttile;
    hold on
    plot(node_out{i}(:,1),phi_true,'-o','Color','r','LineWidth',1.25, 'color', '#009ADE')
    set(gca, ...
        'xgrid','on', ...
        'ygrid','on')
    box on
    title(ax, ['Mode=',num2str((j),'%.0f'),', Freq=',num2str(Freq_true(j),'%.2f'), ' Hz'])
    % set tight layout
    ax = gca;
    xlim([-3.5,3.5])
    ylim([-1.1,1.1])
    xlabel('X (m)')
    ylabel('Y (m)')
end
set(gcf, 'units','centimeters','Position',[10,10,30,4])