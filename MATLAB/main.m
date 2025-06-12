%% modal identification - numerical test
clc
clear
close all

load truss_dataset.mat
fs = 200;

segment_index = 1001:3000;

t = 1/fs*[0:length(acceleration_time_out{1}(1,segment_index))-1];

test_no = 1:20;

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
    [phi,freq,zeta] = AFDD(acceleration_incomplete,t,10, ... % identify more modes for redundancy
        'PickingMethod','auto', ...
        'Ts', 10);

    % SSI
    % [freq,zeta,phi] = SSICOV(acceleration_incomplete,1/fs, ...
    %     'Nmin',2,'Nmax',50);

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