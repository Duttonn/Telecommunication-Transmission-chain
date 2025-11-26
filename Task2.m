%% ========================================================================
%  TASK 2: Robustness Testing Against Noise and Modulation Comparison
%  ========================================================================
%  This script tests the robustness of a communication system against noise
%  and compares performance of different modulation schemes (BPSK, QPSK, 16-QAM)
%  ========================================================================

%% ========================================================================
%  SECTION 0: SETUP
%  ========================================================================
%  Initialize workspace, define simulation parameters, and set random seed
%  for reproducibility.
%  ========================================================================

clear; clc; close all;

% Random seed for reproducibility
rng(42);

% Simulation parameters
num_bits = 1e5;                    % Number of bits to transmit
SNR_range = 0:2:20;                % SNR range in dB
num_SNR = length(SNR_range);

% Modulation orders: BPSK=2, QPSK=4, 16-QAM=16
modulation_orders = [2, 4, 16];
modulation_names = {'BPSK', 'QPSK', '16-QAM'};
num_modulations = length(modulation_orders);

% Impairment parameters
impulsive_prob = 0.02;             % Probability of impulsive noise (2%)
impulsive_amplitude = 5;           % Amplitude of impulsive noise spikes
fading_enabled = true;             % Enable Rayleigh fading

% Display setup info
fprintf('========================================\n');
fprintf('TASK 2: Noise Robustness & Modulation Comparison\n');
fprintf('========================================\n');
fprintf('Number of bits: %d\n', num_bits);
fprintf('SNR range: %d to %d dB\n', min(SNR_range), max(SNR_range));
fprintf('Modulations: BPSK, QPSK, 16-QAM\n');
fprintf('Impulsive noise probability: %.1f%%\n', impulsive_prob*100);
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 1: ADD RANDOM NOISE IN THE CHANNEL
%  ========================================================================
%  Question: Add random noise in the channel:
%            - AWGN (Additive White Gaussian Noise)
%            - Optionally, other impairments (e.g., impulsive noise, fading)
%  
%  This section demonstrates the effect of different channel impairments
%  on the transmitted signal using QPSK modulation as a reference.
%  Visualization includes time-domain waveforms and constellation diagrams.
%  ========================================================================

fprintf('SECTION 1: Visualizing Channel Impairments\n');
fprintf('-------------------------------------------\n');

% Generate random bits and modulate with QPSK for demonstration
num_symbols_demo = 1000;
bits_demo = randi([0 1], num_symbols_demo * 2, 1);  % 2 bits per QPSK symbol
symbols_clean = pskmod(bits_demo, 4, pi/4, 'gray', 'InputType', 'bit');

% --- 1.1 AWGN (Additive White Gaussian Noise) ---
SNR_demo = 10;  % dB for demonstration
symbols_awgn = awgn(symbols_clean, SNR_demo, 'measured');

% --- 1.2 Impulsive Noise (Bernoulli-Gaussian Model) ---
% Random spikes occur with probability impulsive_prob
symbols_impulsive = symbols_clean;
impulse_mask = rand(num_symbols_demo, 1) < impulsive_prob;
impulse_noise = impulsive_amplitude * (randn(num_symbols_demo, 1) + 1j * randn(num_symbols_demo, 1));
symbols_impulsive = symbols_impulsive + impulse_mask .* impulse_noise;

% --- 1.3 Rayleigh Fading ---
% Flat fading: multiply by complex Gaussian (Rayleigh envelope)
rayleigh_coeff = (randn(num_symbols_demo, 1) + 1j * randn(num_symbols_demo, 1)) / sqrt(2);
symbols_fading = symbols_clean .* rayleigh_coeff;

% --- 1.4 Combined: AWGN + Impulsive + Fading ---
symbols_combined = symbols_clean .* rayleigh_coeff;  % Fading first
symbols_combined = symbols_combined + impulse_mask .* impulse_noise;  % Add impulsive
symbols_combined = awgn(symbols_combined, SNR_demo, 'measured');  % Add AWGN

fprintf('Generated %d QPSK symbols for noise visualization\n', num_symbols_demo);
fprintf('AWGN SNR: %d dB\n', SNR_demo);
fprintf('Impulsive noise: %.1f%% probability, amplitude = %d\n', impulsive_prob*100, impulsive_amplitude);
fprintf('Rayleigh fading: enabled\n\n');

% --- Visualization: 2x4 subplot grid ---
% Top row: Time-domain (real part of first 100 symbols)
% Bottom row: Constellation diagrams

figure('Name', 'Section 1: Channel Impairments Visualization', 'Position', [50 50 1400 600]);

plot_range = 1:100;  % Plot first 100 symbols for clarity

% Row 1: Time-domain plots (Real part)
subplot(2,4,1);
plot(plot_range, real(symbols_clean(plot_range)), 'b-', 'LineWidth', 1.2);
title('Clean Signal (Time Domain)');
xlabel('Symbol Index'); ylabel('Real Part');
grid on; ylim([-3 3]);

subplot(2,4,2);
plot(plot_range, real(symbols_awgn(plot_range)), 'r-', 'LineWidth', 1.2);
title('AWGN (Time Domain)');
xlabel('Symbol Index'); ylabel('Real Part');
grid on; ylim([-3 3]);

subplot(2,4,3);
plot(plot_range, real(symbols_impulsive(plot_range)), 'm-', 'LineWidth', 1.2);
title('Impulsive Noise (Time Domain)');
xlabel('Symbol Index'); ylabel('Real Part');
grid on; ylim([-8 8]);

subplot(2,4,4);
plot(plot_range, real(symbols_fading(plot_range)), 'g-', 'LineWidth', 1.2);
title('Rayleigh Fading (Time Domain)');
xlabel('Symbol Index'); ylabel('Real Part');
grid on; ylim([-3 3]);

% Row 2: Constellation diagrams
subplot(2,4,5);
scatter(real(symbols_clean), imag(symbols_clean), 10, 'b', 'filled');
title('Clean Signal (Constellation)');
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);

subplot(2,4,6);
scatter(real(symbols_awgn), imag(symbols_awgn), 10, 'r', 'filled');
title('AWGN (Constellation)');
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);

subplot(2,4,7);
scatter(real(symbols_impulsive), imag(symbols_impulsive), 10, 'm', 'filled');
title('Impulsive Noise (Constellation)');
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
grid on; axis equal; xlim([-10 10]); ylim([-10 10]);

subplot(2,4,8);
scatter(real(symbols_fading), imag(symbols_fading), 10, 'g', 'filled');
title('Rayleigh Fading (Constellation)');
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);

sgtitle('Section 1: Effect of Channel Impairments on QPSK Signal', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 2: MEASURE PERFORMANCE
%  ========================================================================
%  Question: Measure performance:
%            - BER as a function of SNR (draw BER vs. SNR curves)
%  
%  This section measures the Bit Error Rate (BER) as a function of SNR
%  using QPSK modulation, comparing AWGN-only with combined impairments.
%  Theoretical BER curves are included for reference.
%  ========================================================================

fprintf('SECTION 2: Measuring Performance (BER vs SNR)\n');
fprintf('----------------------------------------------\n');

% Initialize BER storage for QPSK
BER_qpsk_awgn = zeros(1, num_SNR);
BER_qpsk_impulsive = zeros(1, num_SNR);
BER_qpsk_fading = zeros(1, num_SNR);
BER_qpsk_combined = zeros(1, num_SNR);

% QPSK parameters
M_qpsk = 4;
k_qpsk = log2(M_qpsk);  % Bits per symbol

% Generate random bits for QPSK simulation
num_symbols = floor(num_bits / k_qpsk);
bits_tx = randi([0 1], num_symbols * k_qpsk, 1);

% Modulate
symbols_tx = pskmod(bits_tx, M_qpsk, pi/4, 'gray', 'InputType', 'bit');

fprintf('Simulating QPSK with %d symbols across SNR range...\n', num_symbols);

for idx = 1:num_SNR
    SNR = SNR_range(idx);
    
    % --- AWGN only ---
    symbols_rx_awgn = awgn(symbols_tx, SNR, 'measured');
    bits_rx_awgn = pskdemod(symbols_rx_awgn, M_qpsk, pi/4, 'gray', 'OutputType', 'bit');
    [~, BER_qpsk_awgn(idx)] = biterr(bits_tx, bits_rx_awgn);
    
    % --- AWGN + Impulsive Noise ---
    symbols_rx_imp = symbols_tx;
    impulse_mask_sim = rand(num_symbols, 1) < impulsive_prob;
    impulse_noise_sim = impulsive_amplitude * (randn(num_symbols, 1) + 1j * randn(num_symbols, 1));
    symbols_rx_imp = symbols_rx_imp + impulse_mask_sim .* impulse_noise_sim;
    symbols_rx_imp = awgn(symbols_rx_imp, SNR, 'measured');
    bits_rx_imp = pskdemod(symbols_rx_imp, M_qpsk, pi/4, 'gray', 'OutputType', 'bit');
    [~, BER_qpsk_impulsive(idx)] = biterr(bits_tx, bits_rx_imp);
    
    % --- Rayleigh Fading + AWGN (no equalization) ---
    rayleigh_coeff_sim = (randn(num_symbols, 1) + 1j * randn(num_symbols, 1)) / sqrt(2);
    symbols_rx_fading = symbols_tx .* rayleigh_coeff_sim;
    symbols_rx_fading = awgn(symbols_rx_fading, SNR, 'measured');
    bits_rx_fading = pskdemod(symbols_rx_fading, M_qpsk, pi/4, 'gray', 'OutputType', 'bit');
    [~, BER_qpsk_fading(idx)] = biterr(bits_tx, bits_rx_fading);
    
    % --- Combined: Fading + Impulsive + AWGN ---
    symbols_rx_comb = symbols_tx .* rayleigh_coeff_sim;
    symbols_rx_comb = symbols_rx_comb + impulse_mask_sim .* impulse_noise_sim;
    symbols_rx_comb = awgn(symbols_rx_comb, SNR, 'measured');
    bits_rx_comb = pskdemod(symbols_rx_comb, M_qpsk, pi/4, 'gray', 'OutputType', 'bit');
    [~, BER_qpsk_combined(idx)] = biterr(bits_tx, bits_rx_comb);
end

% Theoretical BER for QPSK over AWGN
EbNo_linear = 10.^(SNR_range/10) / k_qpsk;  % Convert SNR to Eb/No
BER_theoretical_qpsk = erfc(sqrt(EbNo_linear)) / 2;

fprintf('Simulation complete.\n\n');

% --- Plot BER vs SNR (in percentage) ---
figure('Name', 'Section 2: BER vs SNR Performance', 'Position', [100 100 900 600]);

semilogy(SNR_range, BER_qpsk_awgn * 100, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QPSK - AWGN Only');
hold on;
semilogy(SNR_range, BER_qpsk_impulsive * 100, 'ms-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QPSK - AWGN + Impulsive');
semilogy(SNR_range, BER_qpsk_fading * 100, 'g^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QPSK - Rayleigh Fading + AWGN');
semilogy(SNR_range, BER_qpsk_combined * 100, 'rd-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QPSK - Combined Impairments');
semilogy(SNR_range, BER_theoretical_qpsk * 100, 'k--', 'LineWidth', 2, 'DisplayName', 'QPSK - Theoretical (AWGN)');
hold off;

xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (%)', 'FontSize', 12);
title('Section 2: BER vs SNR for QPSK with Different Channel Impairments', 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 10);
grid on;
ylim([1e-4 100]);
xlim([min(SNR_range) max(SNR_range)]);

%% ========================================================================
%  SECTION 3: COMPARE DIFFERENT MODULATIONS
%  ========================================================================
%  Question: Compare different modulations:
%            - Implement BPSK, QPSK, 16-QAM, etc.
%            - Compare performance at the same SNR
%  
%  This section implements and compares BPSK, QPSK, and 16-QAM modulation
%  schemes over an AWGN channel. BER vs SNR curves and constellation
%  diagrams are plotted for each modulation.
%  ========================================================================

fprintf('SECTION 3: Comparing Different Modulations\n');
fprintf('-------------------------------------------\n');

% Initialize BER storage for all modulations
BER_simulated = zeros(num_modulations, num_SNR);
BER_theoretical = zeros(num_modulations, num_SNR);

% Storage for constellation demo
constellation_symbols = cell(1, num_modulations);
constellation_symbols_noisy = cell(1, num_modulations);

% SNR for constellation visualization (must be in SNR_range: 0:2:20)
SNR_constellation = 14;  % dB

for mod_idx = 1:num_modulations
    M = modulation_orders(mod_idx);
    k = log2(M);  % Bits per symbol
    mod_name = modulation_names{mod_idx};
    
    fprintf('Simulating %s (M=%d, %d bits/symbol)...\n', mod_name, M, k);
    
    % Generate random bits
    num_symbols_mod = floor(num_bits / k);
    bits_tx_mod = randi([0 1], num_symbols_mod * k, 1);
    
    % Modulate based on modulation type
    if M == 2  % BPSK
        symbols_tx_mod = pskmod(bits_tx_mod, M, 0, 'gray', 'InputType', 'bit');
    elseif M == 4  % QPSK
        symbols_tx_mod = pskmod(bits_tx_mod, M, pi/4, 'gray', 'InputType', 'bit');
    else  % 16-QAM
        symbols_tx_mod = qammod(bits_tx_mod, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
    end
    
    % Store clean constellation for visualization
    constellation_symbols{mod_idx} = symbols_tx_mod(1:min(1000, num_symbols_mod));
    
    % Simulate over SNR range
    for snr_idx = 1:num_SNR
        SNR = SNR_range(snr_idx);
        
        % Add AWGN
        symbols_rx_mod = awgn(symbols_tx_mod, SNR, 'measured');
        
        % Store noisy constellation for visualization at specific SNR
        if SNR == SNR_constellation
            constellation_symbols_noisy{mod_idx} = symbols_rx_mod(1:min(1000, num_symbols_mod));
        end
        
        % Demodulate
        if M == 2  % BPSK
            bits_rx_mod = pskdemod(symbols_rx_mod, M, 0, 'gray', 'OutputType', 'bit');
        elseif M == 4  % QPSK
            bits_rx_mod = pskdemod(symbols_rx_mod, M, pi/4, 'gray', 'OutputType', 'bit');
        else  % 16-QAM
            bits_rx_mod = qamdemod(symbols_rx_mod, M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
        end
        
        % Calculate BER
        [~, BER_simulated(mod_idx, snr_idx)] = biterr(bits_tx_mod, bits_rx_mod);
    end
    
    % Calculate theoretical BER
    EbNo_dB = SNR_range - 10*log10(k);  % Convert SNR to Eb/No in dB
    EbNo_linear_mod = 10.^(EbNo_dB/10);
    
    if M == 2  % BPSK theoretical BER
        BER_theoretical(mod_idx, :) = 0.5 * erfc(sqrt(EbNo_linear_mod));
    elseif M == 4  % QPSK theoretical BER (same as BPSK in terms of Eb/No)
        BER_theoretical(mod_idx, :) = 0.5 * erfc(sqrt(EbNo_linear_mod));
    else  % 16-QAM theoretical BER (approximation)
        BER_theoretical(mod_idx, :) = (3/8) * erfc(sqrt((2/5) * EbNo_linear_mod));
    end
end

fprintf('Simulation complete.\n\n');

% --- Plot 1: BER vs SNR Comparison (in percentage) ---
figure('Name', 'Section 3: Modulation Comparison - BER vs SNR', 'Position', [150 150 900 600]);

colors = {'b', 'g', 'r'};
markers = {'o', 's', '^'};

for mod_idx = 1:num_modulations
    % Simulated BER
    semilogy(SNR_range, BER_simulated(mod_idx, :) * 100, ...
        [colors{mod_idx} markers{mod_idx} '-'], 'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', [modulation_names{mod_idx} ' - Simulated']);
    hold on;
    % Theoretical BER
    semilogy(SNR_range, BER_theoretical(mod_idx, :) * 100, ...
        [colors{mod_idx} '--'], 'LineWidth', 2, ...
        'DisplayName', [modulation_names{mod_idx} ' - Theoretical']);
end
hold off;

xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (%)', 'FontSize', 12);
title('Section 3: BER vs SNR for Different Modulation Schemes (AWGN Channel)', 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 9);
grid on;
ylim([1e-4 100]);
xlim([min(SNR_range) max(SNR_range)]);

% --- Plot 2: Constellation Diagrams ---
figure('Name', 'Section 3: Constellation Diagrams', 'Position', [200 100 1200 500]);

for mod_idx = 1:num_modulations
    % Clean constellation
    subplot(2, 3, mod_idx);
    scatter(real(constellation_symbols{mod_idx}), imag(constellation_symbols{mod_idx}), ...
        15, colors{mod_idx}, 'filled');
    title([modulation_names{mod_idx} ' - Clean'], 'FontSize', 12);
    xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
    grid on; axis equal;
    if mod_idx <= 2
        xlim([-2 2]); ylim([-2 2]);
    else
        xlim([-1.5 1.5]); ylim([-1.5 1.5]);
    end
    
    % Noisy constellation
    subplot(2, 3, mod_idx + 3);
    scatter(real(constellation_symbols_noisy{mod_idx}), imag(constellation_symbols_noisy{mod_idx}), ...
        15, colors{mod_idx}, 'filled');
    title([modulation_names{mod_idx} ' - SNR=' num2str(SNR_constellation) 'dB'], 'FontSize', 12);
    xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
    grid on; axis equal;
    if mod_idx <= 2
        xlim([-2 2]); ylim([-2 2]);
    else
        xlim([-1.5 1.5]); ylim([-1.5 1.5]);
    end
end

sgtitle('Section 3: Constellation Diagrams for Different Modulations', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 4: RECAPITULATIVE TABLE
%  ========================================================================
%  Summary comparison of all modulation schemes including:
%  - Bits per symbol (spectral efficiency)
%  - SNR required to achieve target BER (BER = 1e-3)
%  - Performance ranking
%  ========================================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('SECTION 4: RECAPITULATIVE COMPARISON TABLE\n');
fprintf('========================================\n\n');

% Target BER for comparison
target_BER = 1e-3;

% Calculate SNR required for target BER (interpolate from simulation results)
SNR_required = zeros(1, num_modulations);

for mod_idx = 1:num_modulations
    ber_curve = BER_simulated(mod_idx, :);
    
    % Find where BER crosses target (interpolate)
    idx_below = find(ber_curve <= target_BER, 1, 'first');
    
    if isempty(idx_below)
        SNR_required(mod_idx) = NaN;  % Target BER not achieved
    elseif idx_below == 1
        SNR_required(mod_idx) = SNR_range(1);
    else
        % Linear interpolation in log domain
        idx_above = idx_below - 1;
        snr_low = SNR_range(idx_above);
        snr_high = SNR_range(idx_below);
        ber_low = ber_curve(idx_above);
        ber_high = ber_curve(idx_below);
        
        % Interpolate
        log_target = log10(target_BER);
        log_low = log10(ber_low);
        log_high = log10(ber_high);
        
        SNR_required(mod_idx) = snr_low + (snr_high - snr_low) * ...
            (log_target - log_low) / (log_high - log_low);
    end
end

% Spectral efficiency (bits per symbol)
bits_per_symbol = log2(modulation_orders);

% Create and display table
fprintf('%-12s | %-15s | %-20s | %-20s\n', ...
    'Modulation', 'Bits/Symbol', 'SNR @ BER=1e-3 (dB)', 'Spectral Eff.');
fprintf('%-12s-+-%-15s-+-%-20s-+-%-20s\n', ...
    '------------', '---------------', '--------------------', '--------------------');

for mod_idx = 1:num_modulations
    if isnan(SNR_required(mod_idx))
        snr_str = 'Not achieved';
    else
        snr_str = sprintf('%.2f', SNR_required(mod_idx));
    end
    
    fprintf('%-12s | %-15d | %-20s | %-20s\n', ...
        modulation_names{mod_idx}, ...
        bits_per_symbol(mod_idx), ...
        snr_str, ...
        sprintf('%d bits/symbol', bits_per_symbol(mod_idx)));
end

fprintf('\n');

% Performance summary
fprintf('========================================\n');
fprintf('PERFORMANCE SUMMARY\n');
fprintf('========================================\n');
fprintf('- BPSK: Most robust, lowest spectral efficiency (1 bit/symbol)\n');
fprintf('- QPSK: Good balance of robustness and efficiency (2 bits/symbol)\n');
fprintf('- 16-QAM: Highest efficiency but requires higher SNR (4 bits/symbol)\n');
fprintf('\n');
fprintf('RECOMMENDATION:\n');
fprintf('- Low SNR environments: Use BPSK or QPSK\n');
fprintf('- High SNR environments: Use 16-QAM for better throughput\n');
fprintf('========================================\n');

% Create MATLAB table for additional display
summary_table = table(modulation_names', bits_per_symbol', SNR_required', ...
    'VariableNames', {'Modulation', 'BitsPerSymbol', 'SNR_at_BER_1e3_dB'});

fprintf('\nMATLAB Table Format:\n');
disp(summary_table);

fprintf('\n*** Task 2 Complete ***\n');
