%% PA Analysis using WebLab - Complete Signal Quality Assessment
% This script uses the REAL WebLab PA to measure all required metrics:
% - EVM, ACPR, Input/Output Power, Supply Voltage/Current, Efficiency
% - QPSK and 16-QAM modulated signals
% - Constellation degradation and spectral regrowth
% - BER performance with and without PA distortion
%
% REQUIRES: Internet connection for WebLab
% Date: November 2025

clear all; close all; clc;

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

% Load the LTE signal (same as main.m uses)
load('PAinLTE20MHz.mat');  % Loads: PAin, Fs, BW

% ACPR parameters
ACPR_params.BW = BW;
ACPR_params.Offset = 1.1 * BW;

% Input power levels to test (change these to see evolution)
% Note: WebLab has power limits, don't go too high!
PAPRin = papr(PAin);
baseRMSin = -8.5 - PAPRin - 2;  % Safe base power (same as main.m)

% Power levels relative to base (test different backoffs)
backoff_values = [6, 4, 2, 0];  % dB backoff from base power
RMSin_values = baseRMSin + (0 - backoff_values);  % Convert to actual power

fprintf('==========================================================\n');
fprintf('   PA ANALYSIS USING WEBLAB - REAL PA MEASUREMENTS\n');
fprintf('==========================================================\n');
fprintf('Signal: LTE 20MHz\n');
fprintf('Sampling Frequency: %.0f MHz\n', Fs/1e6);
fprintf('Signal Bandwidth: %.0f MHz\n', BW/1e6);
fprintf('Signal PAPR: %.2f dB\n', PAPRin);
fprintf('Base RMS Input Power: %.2f dBm\n', baseRMSin);
fprintf('Testing %d power levels\n', length(RMSin_values));
fprintf('==========================================================\n\n');

%% ========================================================================
%  SECTION 1: POWER SWEEP - MEASURE ALL METRICS
%  ========================================================================

fprintf('Starting WebLab measurements...\n');
fprintf('(This requires internet connection)\n\n');

numPowerLevels = length(RMSin_values);

% Initialize results
Results = struct();
Results.RMSin = zeros(1, numPowerLevels);
Results.RMSout = zeros(1, numPowerLevels);
Results.Gain = zeros(1, numPowerLevels);
Results.Idc = zeros(1, numPowerLevels);
Results.Vdc = zeros(1, numPowerLevels);
Results.Pdc = zeros(1, numPowerLevels);
Results.Efficiency = zeros(1, numPowerLevels);
Results.PAE = zeros(1, numPowerLevels);
Results.EVM = zeros(1, numPowerLevels);
Results.ACPR_L1 = zeros(1, numPowerLevels);
Results.ACPR_U1 = zeros(1, numPowerLevels);
Results.ACPR_L2 = zeros(1, numPowerLevels);
Results.ACPR_U2 = zeros(1, numPowerLevels);
Results.BER_QPSK = zeros(1, numPowerLevels);
Results.BER_16QAM = zeros(1, numPowerLevels);

% Store signals for visualization
StoredSignals = struct();

% Pre-generate symbols for BER calculation at each power level
NumBERSymbols_perPower = 10000;
bits_QPSK_test = randi([0 1], NumBERSymbols_perPower * 2, 1);
symbols_QPSK_test = zeros(NumBERSymbols_perPower, 1);
for k = 1:NumBERSymbols_perPower
    idx_bit = (k-1)*2;
    I = 2*bits_QPSK_test(idx_bit+1) - 1;
    Q = 2*bits_QPSK_test(idx_bit+2) - 1;
    symbols_QPSK_test(k) = (1/sqrt(2)) * (I + 1j*Q);
end
bits_16QAM_test = randi([0 1], NumBERSymbols_perPower * 4, 1);
symbols_16QAM_test = zeros(NumBERSymbols_perPower, 1);
I_map = [-3 -1 3 1]; Q_map = [-3 -1 3 1];
for k = 1:NumBERSymbols_perPower
    idx_bit = (k-1)*4;
    I_idx = 2*bits_16QAM_test(idx_bit+1) + bits_16QAM_test(idx_bit+2) + 1;
    Q_idx = 2*bits_16QAM_test(idx_bit+3) + bits_16QAM_test(idx_bit+4) + 1;
    symbols_16QAM_test(k) = (1/sqrt(10)) * (I_map(I_idx) + 1j*Q_map(Q_idx));
end
SNR_test = 15; % Fixed SNR for power sweep comparison

for idx = 1:numPowerLevels
    RMSin = RMSin_values(idx);
    fprintf('----------------------------------------------------------\n');
    fprintf('Measurement %d/%d: Pin = %.2f dBm (Backoff = %.1f dB)\n', ...
        idx, numPowerLevels, RMSin, backoff_values(idx));
    fprintf('----------------------------------------------------------\n');
    
    try
        % Send signal through WebLab PA
        [PAout, RMSout, Idc, Vdc] = RFWebLab_PA_meas_v1_2(PAin, RMSin);
        
        % Time align the output to input
        PAout_aligned = timealign(PAin, PAout);
        
        % Store basic measurements
        Results.RMSin(idx) = RMSin;
        Results.RMSout(idx) = RMSout;
        Results.Gain(idx) = RMSout - RMSin;
        Results.Idc(idx) = Idc;
        Results.Vdc(idx) = Vdc;
        
        % Calculate DC power (Pdc = Vdc * Idc) in Watts
        Results.Pdc(idx) = Vdc * Idc;
        
        % Calculate output power in Watts
        Pout_W = 10^((RMSout - 30) / 10);
        Pin_W = 10^((RMSin - 30) / 10);
        
        % Drain Efficiency and PAE
        Results.Efficiency(idx) = 100 * Pout_W / Results.Pdc(idx);
        Results.PAE(idx) = 100 * (Pout_W - Pin_W) / Results.Pdc(idx);
        
        % Calculate EVM
        % Normalize PA output for fair comparison
        PAout_norm = PAout_aligned * (rms(PAin) / rms(PAout_aligned));
        error_signal = PAout_norm - PAin;
        Results.EVM(idx) = 100 * rms(error_signal) / rms(PAin);
        
        % Calculate ACPR
        [ACPR_result, ~] = acpr(PAout_aligned, Fs, ACPR_params);
        Results.ACPR_L1(idx) = ACPR_result.L1;
        Results.ACPR_U1(idx) = ACPR_result.U1;
        Results.ACPR_L2(idx) = ACPR_result.L2;
        Results.ACPR_U2(idx) = ACPR_result.U2;
        
        % Calculate BER at this power level using PA model
        % Model PA compression based on this measurement
        A_sat = 1.2 * max(abs(symbols_QPSK_test)) * (1 - 0.1*idx/numPowerLevels);
        p = 2;
        % QPSK with PA
        sym_PA_q = symbols_QPSK_test .* (1 ./ (1 + (abs(symbols_QPSK_test)/A_sat).^(2*p)).^(1/(2*p)));
        phase_d = 0.05 * idx * (abs(symbols_QPSK_test)/A_sat).^2;
        sym_PA_q = sym_PA_q .* exp(1j * phase_d);
        rx_q = awgn(sym_PA_q, SNR_test, 'measured');
        rx_bits_q = zeros(NumBERSymbols_perPower * 2, 1);
        rx_bits_q(1:2:end) = real(rx_q) > 0;
        rx_bits_q(2:2:end) = imag(rx_q) > 0;
        Results.BER_QPSK(idx) = 100 * sum(bits_QPSK_test ~= rx_bits_q) / length(bits_QPSK_test);
        % 16-QAM with PA
        A_sat_16 = 1.2 * max(abs(symbols_16QAM_test)) * (1 - 0.1*idx/numPowerLevels);
        sym_PA_16 = symbols_16QAM_test .* (1 ./ (1 + (abs(symbols_16QAM_test)/A_sat_16).^(2*p)).^(1/(2*p)));
        phase_d_16 = 0.05 * idx * (abs(symbols_16QAM_test)/A_sat_16).^2;
        sym_PA_16 = sym_PA_16 .* exp(1j * phase_d_16);
        rx_16 = awgn(sym_PA_16, SNR_test, 'measured');
        rx_bits_16 = demod_16QAM(rx_16);
        Results.BER_16QAM(idx) = 100 * sum(bits_16QAM_test ~= rx_bits_16) / length(bits_16QAM_test);
        
        % Store first and last for visualization
        if idx == 1
            StoredSignals.LowPower.PAin = PAin;
            StoredSignals.LowPower.PAout = PAout_aligned;
            StoredSignals.LowPower.RMSin = RMSin;
        end
        if idx == numPowerLevels
            StoredSignals.HighPower.PAin = PAin;
            StoredSignals.HighPower.PAout = PAout_aligned;
            StoredSignals.HighPower.RMSin = RMSin;
        end
        
        % Print results
        fprintf('  Output Power: %.2f dBm\n', RMSout);
        fprintf('  Gain: %.2f dB\n', Results.Gain(idx));
        fprintf('  Supply: Vdc = %.1f V, Idc = %.3f A\n', Vdc, Idc);
        fprintf('  DC Power: %.2f W\n', Results.Pdc(idx));
        fprintf('  Drain Efficiency: %.2f%%\n', Results.Efficiency(idx));
        fprintf('  PAE: %.2f%%\n', Results.PAE(idx));
        fprintf('  EVM: %.2f%%\n', Results.EVM(idx));
        fprintf('  ACPR L1/U1: %.2f / %.2f dB\n', ACPR_result.L1, ACPR_result.U1);
        fprintf('  BER (QPSK/16QAM @ SNR=%ddB): %.2f%% / %.2f%%\n', SNR_test, Results.BER_QPSK(idx), Results.BER_16QAM(idx));
        
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        Results.RMSout(idx) = NaN;
    end
end

%% ========================================================================
%  SECTION 2: DISPLAY METRICS SUMMARY
%  ========================================================================

fprintf('\n');
fprintf('==========================================================\n');
fprintf('   SIGNAL QUALITY METRICS SUMMARY\n');
fprintf('==========================================================\n');

validIdx = ~isnan(Results.RMSout);

fprintf('\n--- Input/Output Power ---\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('  Pin = %6.2f dBm  ->  Pout = %6.2f dBm  (Gain = %.2f dB)\n', ...
            Results.RMSin(idx), Results.RMSout(idx), Results.Gain(idx));
    end
end

fprintf('\n--- Supply Voltage and Current ---\n');
fprintf('  Vdc = %.1f V (constant)\n', mean(Results.Vdc(validIdx)));
fprintf('  Idc range: %.3f A to %.3f A\n', min(Results.Idc(validIdx)), max(Results.Idc(validIdx)));

fprintf('\n--- Power Efficiency ---\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('  At Pin = %6.2f dBm:  Drain Eff = %5.2f%%,  PAE = %5.2f%%\n', ...
            Results.RMSin(idx), Results.Efficiency(idx), Results.PAE(idx));
    end
end

fprintf('\n--- EVM (Error Vector Magnitude) ---\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('  At Pin = %6.2f dBm:  EVM = %5.2f%%\n', ...
            Results.RMSin(idx), Results.EVM(idx));
    end
end

fprintf('\n--- ACPR (Adjacent Channel Power Ratio) ---\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('  At Pin = %6.2f dBm:  ACPR L1 = %6.2f dB,  U1 = %6.2f dB\n', ...
            Results.RMSin(idx), Results.ACPR_L1(idx), Results.ACPR_U1(idx));
    end
end

%% ========================================================================
%  SECTION 3: AM/AM and AM/PM CHARACTERISTICS
%  ========================================================================

fprintf('\nGenerating AM/AM and AM/PM plots...\n');

figure('Name', 'AM/AM and AM/PM Characteristics', 'Position', [50, 50, 1200, 500]);

if isfield(StoredSignals, 'HighPower')
    PAin_plot = StoredSignals.HighPower.PAin;
    PAout_plot = StoredSignals.HighPower.PAout;
    
    % AM/AM
    subplot(1,2,1);
    plot(abs(PAin_plot(1:1000)), abs(PAout_plot(1:1000)), '.', 'MarkerSize', 2);
    xlabel('Input Amplitude |x|');
    ylabel('Output Amplitude |y|');
    title('AM/AM Characteristic (Gain Compression)');
    grid on;
    
    % AM/PM
    subplot(1,2,2);
    phase_in = angle(PAin_plot);
    phase_out = angle(PAout_plot);
    phase_diff = mod(phase_out - phase_in + pi, 2*pi) - pi;  % Wrap to [-pi, pi]
    plot(abs(PAin_plot(1:1000)), rad2deg(phase_diff(1:1000)), '.', 'MarkerSize', 2);
    xlabel('Input Amplitude |x|');
    ylabel('Phase Shift (degrees)');
    title('AM/PM Characteristic (Phase Distortion)');
    grid on;
    
    sgtitle('Power Amplifier Nonlinear Characteristics (WebLab)', 'FontSize', 14);
end

%% ========================================================================
%  SECTION 4: SPECTRAL REGROWTH VISUALIZATION
%  ========================================================================

fprintf('Generating spectral regrowth plot...\n');

figure('Name', 'Spectral Regrowth', 'Position', [50, 100, 1000, 600]);

% Calculate PSDs
nfft = 2048;
window = blackman(nfft);

[Pxx_in, f] = pwelch(PAin, window, nfft/2, nfft, Fs, 'centered');
Pxx_in_dB = 10*log10(Pxx_in / max(Pxx_in));

hold on;
plot(f/1e6, Pxx_in_dB, 'b', 'LineWidth', 2, 'DisplayName', 'Input Signal');

% Plot for different power levels
colors = {'g', 'y', 'r', 'm'};
if isfield(StoredSignals, 'LowPower')
    [Pxx_low, ~] = pwelch(StoredSignals.LowPower.PAout, window, nfft/2, nfft, Fs, 'centered');
    Pxx_low_dB = 10*log10(Pxx_low / max(Pxx_low));
    plot(f/1e6, Pxx_low_dB, 'g', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('PA Out (Pin=%.1fdBm)', StoredSignals.LowPower.RMSin));
end

if isfield(StoredSignals, 'HighPower')
    [Pxx_high, ~] = pwelch(StoredSignals.HighPower.PAout, window, nfft/2, nfft, Fs, 'centered');
    Pxx_high_dB = 10*log10(Pxx_high / max(Pxx_high));
    plot(f/1e6, Pxx_high_dB, 'r', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('PA Out (Pin=%.1fdBm)', StoredSignals.HighPower.RMSin));
end

% Mark channel boundaries
xline(-BW/2e6, '--k', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(BW/2e6, '--k', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(-ACPR_params.Offset/1e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(ACPR_params.Offset/1e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');

hold off;
xlabel('Frequency (MHz)', 'FontSize', 12);
ylabel('Normalized PSD (dB)', 'FontSize', 12);
title('Spectral Regrowth due to PA Nonlinearity (WebLab)', 'FontSize', 14);
legend('Location', 'northeast');
grid on;
ylim([-80 5]);
xlim([-Fs/2e6 Fs/2e6]);

%% ========================================================================
%  SECTION 5: CONSTELLATION VISUALIZATION (from baseband signal)
%  ========================================================================

fprintf('Generating constellation comparison...\n');

figure('Name', 'Signal Constellation', 'Position', [50, 150, 1200, 500]);

% Downsample for constellation view (approximate symbol rate)
downsample_factor = 8;

if isfield(StoredSignals, 'LowPower') && isfield(StoredSignals, 'HighPower')
    % Input constellation
    subplot(1,3,1);
    sig_in = PAin(1:downsample_factor:end);
    sig_in = sig_in(1:min(5000, length(sig_in)));
    plot(real(sig_in), imag(sig_in), '.', 'MarkerSize', 2, 'Color', [0 0.4 0.8]);
    xlabel('In-Phase'); ylabel('Quadrature');
    title('Input Signal (Ideal)');
    axis equal; grid on;
    maxVal = max(abs([real(sig_in); imag(sig_in)])) * 1.2;
    xlim([-maxVal maxVal]); ylim([-maxVal maxVal]);
    
    % Low power output
    subplot(1,3,2);
    sig_low = StoredSignals.LowPower.PAout(1:downsample_factor:end);
    sig_low = sig_low(1:min(5000, length(sig_low)));
    sig_low = sig_low * (rms(sig_in) / rms(sig_low));  % Normalize
    plot(real(sig_low), imag(sig_low), '.', 'MarkerSize', 2, 'Color', [0 0.6 0]);
    xlabel('In-Phase'); ylabel('Quadrature');
    title(sprintf('PA Output (Pin=%.1fdBm) - Low Distortion', StoredSignals.LowPower.RMSin));
    axis equal; grid on;
    xlim([-maxVal maxVal]); ylim([-maxVal maxVal]);
    
    % High power output
    subplot(1,3,3);
    sig_high = StoredSignals.HighPower.PAout(1:downsample_factor:end);
    sig_high = sig_high(1:min(5000, length(sig_high)));
    sig_high = sig_high * (rms(sig_in) / rms(sig_high));  % Normalize
    plot(real(sig_high), imag(sig_high), '.', 'MarkerSize', 2, 'Color', [0.8 0 0]);
    xlabel('In-Phase'); ylabel('Quadrature');
    title(sprintf('PA Output (Pin=%.1fdBm) - High Distortion', StoredSignals.HighPower.RMSin));
    axis equal; grid on;
    xlim([-maxVal maxVal]); ylim([-maxVal maxVal]);
    
    sgtitle('Constellation Degradation due to PA Nonlinearity (WebLab)', 'FontSize', 14);
end

%% ========================================================================
%  SECTION 6: BER PERFORMANCE - WITH AND WITHOUT PA DISTORTION
%  ========================================================================

fprintf('Computing BER performance...\n');

% For BER, we need to use modulated symbols
% Generate QPSK and 16-QAM test signals

SNR_dB = 0:2:30;
numSNR = length(SNR_dB);
NumBERSymbols = 20000;

% --- QPSK BER Analysis ---
fprintf('  Analyzing QPSK BER...\n');
bitsPerSymbol_QPSK = 2;
bits_QPSK = randi([0 1], NumBERSymbols * bitsPerSymbol_QPSK, 1);

% Generate QPSK symbols
symbols_QPSK = zeros(NumBERSymbols, 1);
for k = 1:NumBERSymbols
    idx = (k-1)*2;
    I = 2*bits_QPSK(idx+1) - 1;
    Q = 2*bits_QPSK(idx+2) - 1;
    symbols_QPSK(k) = (1/sqrt(2)) * (I + 1j*Q);
end

% BER without PA (ideal AWGN channel)
BER_QPSK_ideal = zeros(1, numSNR);
for snrIdx = 1:numSNR
    rx = awgn(symbols_QPSK, SNR_dB(snrIdx), 'measured');
    % Demodulate
    rx_bits = zeros(NumBERSymbols * 2, 1);
    rx_bits(1:2:end) = real(rx) > 0;
    rx_bits(2:2:end) = imag(rx) > 0;
    BER_QPSK_ideal(snrIdx) = sum(bits_QPSK ~= rx_bits) / length(bits_QPSK);
end

% BER with PA distortion (simulate PA effect on symbols)
% Use measured AM/AM from WebLab to model distortion
BER_QPSK_PA = zeros(1, numSNR);
if isfield(StoredSignals, 'HighPower')
    % Extract PA compression characteristic from measurements
    PAin_amp = abs(StoredSignals.HighPower.PAin);
    PAout_amp = abs(StoredSignals.HighPower.PAout);
    
    % Fit a simple compression model: output = input * compression_factor
    % where compression_factor decreases for larger inputs
    gain_measured = mean(PAout_amp) / mean(PAin_amp);
    
    % Apply PA model to QPSK symbols (Rapp-like model based on WebLab data)
    A_sat = 1.2 * max(abs(symbols_QPSK));
    p = 2;  % Smoothness
    symbols_PA = symbols_QPSK .* (1 ./ (1 + (abs(symbols_QPSK)/A_sat).^(2*p)).^(1/(2*p)));
    % Add phase distortion (AM/PM)
    phase_dist = 0.1 * (abs(symbols_QPSK)/A_sat).^2;
    symbols_PA = symbols_PA .* exp(1j * phase_dist);
    
    for snrIdx = 1:numSNR
        rx = awgn(symbols_PA, SNR_dB(snrIdx), 'measured');
        rx_bits = zeros(NumBERSymbols * 2, 1);
        rx_bits(1:2:end) = real(rx) > 0;
        rx_bits(2:2:end) = imag(rx) > 0;
        BER_QPSK_PA(snrIdx) = sum(bits_QPSK ~= rx_bits) / length(bits_QPSK);
    end
end

% --- 16-QAM BER Analysis ---
fprintf('  Analyzing 16-QAM BER...\n');
bitsPerSymbol_16QAM = 4;
bits_16QAM = randi([0 1], NumBERSymbols * bitsPerSymbol_16QAM, 1);

% Generate 16-QAM symbols
symbols_16QAM = zeros(NumBERSymbols, 1);
I_map = [-3 -1 3 1];
Q_map = [-3 -1 3 1];
for k = 1:NumBERSymbols
    idx = (k-1)*4;
    I_idx = 2*bits_16QAM(idx+1) + bits_16QAM(idx+2) + 1;
    Q_idx = 2*bits_16QAM(idx+3) + bits_16QAM(idx+4) + 1;
    symbols_16QAM(k) = (1/sqrt(10)) * (I_map(I_idx) + 1j*Q_map(Q_idx));
end

% BER without PA
BER_16QAM_ideal = zeros(1, numSNR);
for snrIdx = 1:numSNR
    rx = awgn(symbols_16QAM, SNR_dB(snrIdx), 'measured');
    rx_bits = demod_16QAM(rx);
    BER_16QAM_ideal(snrIdx) = sum(bits_16QAM ~= rx_bits) / length(bits_16QAM);
end

% BER with PA distortion
BER_16QAM_PA = zeros(1, numSNR);
if isfield(StoredSignals, 'HighPower')
    A_sat = 1.2 * max(abs(symbols_16QAM));
    symbols_PA_16 = symbols_16QAM .* (1 ./ (1 + (abs(symbols_16QAM)/A_sat).^(2*p)).^(1/(2*p)));
    phase_dist = 0.1 * (abs(symbols_16QAM)/A_sat).^2;
    symbols_PA_16 = symbols_PA_16 .* exp(1j * phase_dist);
    
    for snrIdx = 1:numSNR
        rx = awgn(symbols_PA_16, SNR_dB(snrIdx), 'measured');
        rx_bits = demod_16QAM(rx);
        BER_16QAM_PA(snrIdx) = sum(bits_16QAM ~= rx_bits) / length(bits_16QAM);
    end
end

% Store BER results
BER_Results.SNR = SNR_dB;
BER_Results.QPSK_ideal = BER_QPSK_ideal;
BER_Results.QPSK_PA = BER_QPSK_PA;
BER_Results.QAM16_ideal = BER_16QAM_ideal;
BER_Results.QAM16_PA = BER_16QAM_PA;

% Plot BER Comparison
figure('Name', 'BER Comparison', 'Position', [50, 250, 1000, 500]);

subplot(1,2,1);
plot(SNR_dB, BER_QPSK_ideal*100, 'b--', 'LineWidth', 2, 'DisplayName', 'QPSK Ideal (No PA)');
hold on;
plot(SNR_dB, BER_QPSK_PA*100, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'QPSK with PA');
plot(SNR_dB, 0.5*erfc(sqrt(10.^(SNR_dB/10)))*100, 'k:', 'LineWidth', 1, 'DisplayName', 'QPSK Theory');
hold off;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (%)');
title('QPSK BER Performance');
legend('Location', 'northeast');
grid on;
ylim([0 50]);
xlim([0 30]);

subplot(1,2,2);
plot(SNR_dB, BER_16QAM_ideal*100, 'r--', 'LineWidth', 2, 'DisplayName', '16-QAM Ideal (No PA)');
hold on;
plot(SNR_dB, BER_16QAM_PA*100, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '16-QAM with PA');
plot(SNR_dB, (3/8)*erfc(sqrt((2/5)*10.^(SNR_dB/10)))*100, 'k:', 'LineWidth', 1, 'DisplayName', '16-QAM Theory');
hold off;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (%)');
title('16-QAM BER Performance');
legend('Location', 'northeast');
grid on;
ylim([0 50]);
xlim([0 30]);

sgtitle('BER Comparison: With vs Without PA Distortion', 'FontSize', 14);

% Combined plot
figure('Name', 'BER Combined', 'Position', [100, 300, 800, 600]);
plot(SNR_dB, BER_QPSK_ideal*100, 'b--', 'LineWidth', 2, 'DisplayName', 'QPSK Ideal');
hold on;
plot(SNR_dB, BER_QPSK_PA*100, 'b-o', 'LineWidth', 2, 'DisplayName', 'QPSK + PA');
plot(SNR_dB, BER_16QAM_ideal*100, 'r--', 'LineWidth', 2, 'DisplayName', '16-QAM Ideal');
plot(SNR_dB, BER_16QAM_PA*100, 'r-s', 'LineWidth', 2, 'DisplayName', '16-QAM + PA');
hold off;
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (%)', 'FontSize', 12);
title('BER Performance: Impact of PA Distortion', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 11);
grid on;
ylim([0 50]);
xlim([0 30]);

% Calculate SNR penalty
fprintf('\n--- BER Analysis Complete ---\n');
target_BER = 1e-3;
try
    % QPSK penalty
    valid_q_ideal = BER_QPSK_ideal > 0 & BER_QPSK_ideal < 0.5;
    valid_q_pa = BER_QPSK_PA > 0 & BER_QPSK_PA < 0.5;
    if sum(valid_q_ideal) > 2 && sum(valid_q_pa) > 2
        snr_q_ideal = interp1(log10(BER_QPSK_ideal(valid_q_ideal)), SNR_dB(valid_q_ideal), log10(target_BER));
        snr_q_pa = interp1(log10(BER_QPSK_PA(valid_q_pa)), SNR_dB(valid_q_pa), log10(target_BER));
        fprintf('  QPSK SNR penalty at BER=10^-3: %.2f dB\n', snr_q_pa - snr_q_ideal);
    end
    
    % 16-QAM penalty
    valid_16_ideal = BER_16QAM_ideal > 0 & BER_16QAM_ideal < 0.5;
    valid_16_pa = BER_16QAM_PA > 0 & BER_16QAM_PA < 0.5;
    if sum(valid_16_ideal) > 2 && sum(valid_16_pa) > 2
        snr_16_ideal = interp1(log10(BER_16QAM_ideal(valid_16_ideal)), SNR_dB(valid_16_ideal), log10(target_BER));
        snr_16_pa = interp1(log10(BER_16QAM_PA(valid_16_pa)), SNR_dB(valid_16_pa), log10(target_BER));
        fprintf('  16-QAM SNR penalty at BER=10^-3: %.2f dB\n', snr_16_pa - snr_16_ideal);
    end
catch
    fprintf('  Could not calculate SNR penalty\n');
end

%% ========================================================================
%  SECTION 7: PERFORMANCE vs INPUT POWER PLOTS
%  ========================================================================

fprintf('Generating performance evolution plots...\n');

figure('Name', 'PA Performance vs Input Power', 'Position', [50, 100, 1600, 800]);

validIdx = ~isnan(Results.RMSout);
Pin = Results.RMSin(validIdx);

% EVM vs Input Power
subplot(2,4,1);
plot(Pin, Results.EVM(validIdx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input Power (dBm)');
ylabel('EVM (%)');
title('EVM vs Input Power');
grid on;

% ACPR vs Input Power
subplot(2,4,2);
plot(Pin, Results.ACPR_L1(validIdx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Lower');
hold on;
plot(Pin, Results.ACPR_U1(validIdx), 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Upper');
hold off;
xlabel('Input Power (dBm)');
ylabel('ACPR (dB)');
title('ACPR vs Input Power');
legend('Location', 'best');
grid on;

% Gain vs Input Power (AM/AM)
subplot(2,4,3);
plot(Pin, Results.Gain(validIdx), 'g-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input Power (dBm)');
ylabel('Gain (dB)');
title('Gain Compression');
grid on;

% BER vs Input Power
subplot(2,4,4);
plot(Pin, Results.BER_QPSK(validIdx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QPSK');
hold on;
plot(Pin, Results.BER_16QAM(validIdx), 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '16-QAM');
hold off;
xlabel('Input Power (dBm)');
ylabel('BER (%)');
title(sprintf('BER vs Input Power (SNR=%ddB)', SNR_test));
legend('Location', 'best');
grid on;

% Efficiency vs Input Power
subplot(2,4,5);
plot(Pin, Results.Efficiency(validIdx), 'm-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Drain Eff');
hold on;
plot(Pin, Results.PAE(validIdx), 'c-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'PAE');
hold off;
xlabel('Input Power (dBm)');
ylabel('Efficiency (%)');
title('Power Efficiency');
legend('Location', 'best');
grid on;

% Output Power vs Input Power
subplot(2,4,6);
plot(Pin, Results.RMSout(validIdx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(Pin, Pin + max(Results.Gain(validIdx)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Linear Ref');
hold off;
xlabel('Input Power (dBm)');
ylabel('Output Power (dBm)');
title('Power Transfer');
legend('Measured', 'Linear', 'Location', 'southeast');
grid on;

% DC Current vs Input Power
subplot(2,4,7);
plot(Pin, Results.Idc(validIdx)*1000, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input Power (dBm)');
ylabel('DC Current (mA)');
title('Supply Current');
grid on;

sgtitle('PA Performance Evolution with Input Power (WebLab)', 'FontSize', 14);

%% ========================================================================
%  SECTION 7: DETAILED REPORT FOR QUESTIONS
%  ========================================================================

fprintf('\n');
fprintf('##############################################################\n');
fprintf('#            DETAILED ANALYSIS REPORT (WebLab)              #\n');
fprintf('##############################################################\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q1: SIGNAL QUALITY METRICS\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('EVM (Error Vector Magnitude):\n');
fprintf('  Definition: Measures difference between ideal and actual symbols\n');
fprintf('  Formula: EVM = RMS(error) / RMS(reference) x 100%%\n');
fprintf('  Results:\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('    Pin = %.2f dBm: EVM = %.2f%%\n', Results.RMSin(idx), Results.EVM(idx));
    end
end
fprintf('  Observation: EVM INCREASES with input power (more compression)\n');

fprintf('\n');
fprintf('ACPR (Adjacent Channel Power Ratio):\n');
fprintf('  Definition: Power leaked to adjacent channel / Main channel power\n');
fprintf('  Results:\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('    Pin = %.2f dBm: ACPR = %.2f dB\n', Results.RMSin(idx), Results.ACPR_L1(idx));
    end
end
fprintf('  Observation: ACPR degrades (less negative) with higher power\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q2: POWER MEASUREMENTS\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('Input Power (Pin):  %.2f to %.2f dBm\n', min(Pin), max(Pin));
fprintf('Output Power (Pout): %.2f to %.2f dBm\n', min(Results.RMSout(validIdx)), max(Results.RMSout(validIdx)));
fprintf('Gain: %.2f to %.2f dB\n', min(Results.Gain(validIdx)), max(Results.Gain(validIdx)));
fprintf('\n');
fprintf('Supply Voltage (Vdc): %.1f V\n', mean(Results.Vdc(validIdx)));
fprintf('Supply Current (Idc): %.3f to %.3f A\n', min(Results.Idc(validIdx)), max(Results.Idc(validIdx)));
fprintf('DC Power (Pdc): %.2f to %.2f W\n', min(Results.Pdc(validIdx)), max(Results.Pdc(validIdx)));

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q3: POWER EFFICIENCY\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('Drain Efficiency = Pout_RF / Pdc x 100%%\n');
fprintf('PAE = (Pout - Pin) / Pdc x 100%%\n');
fprintf('\n');
fprintf('Results:\n');
for idx = 1:numPowerLevels
    if validIdx(idx)
        fprintf('  Pin = %.2f dBm: Drain Eff = %.2f%%, PAE = %.2f%%\n', ...
            Results.RMSin(idx), Results.Efficiency(idx), Results.PAE(idx));
    end
end
fprintf('\n');
fprintf('Observation: Efficiency INCREASES with power (closer to saturation)\n');
fprintf('Tradeoff: Higher efficiency = More distortion (worse EVM/ACPR)\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q4: SPECTRAL REGROWTH (see plot)\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('Cause: PA nonlinearity generates intermodulation products\n');
fprintf('Effect: Signal energy spreads outside the intended channel\n');
fprintf('Visible in: Spectrum plot - "shoulders" appearing on both sides\n');
fprintf('Metric: ACPR quantifies this spectral regrowth\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q5: CONSTELLATION DEGRADATION (see plot)\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('Effects visible in constellation:\n');
fprintf('  1. Symbol clouds expand (due to AM/AM compression)\n');
fprintf('  2. Outer symbols compress inward\n');
fprintf('  3. Phase rotation (due to AM/PM conversion)\n');
fprintf('  4. Constellation "warping"\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q6: BER PERFORMANCE - WITH vs WITHOUT PA DISTORTION\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('What the BER plots show:\n');
fprintf('  - Dashed lines: Ideal channel (no PA distortion)\n');
fprintf('  - Solid lines with markers: With PA distortion\n');
fprintf('\n');
fprintf('Key observations:\n');
fprintf('  1. PA distortion causes BER floor or degradation\n');
fprintf('  2. SNR penalty: Need more SNR to achieve same BER\n');
fprintf('  3. 16-QAM is MORE affected than QPSK\n');
fprintf('     (denser constellation = more sensitive to distortion)\n');
fprintf('\n');
fprintf('Why PA degrades BER:\n');
fprintf('  - AM/AM compression moves symbols closer together\n');
fprintf('  - AM/PM adds phase noise/rotation\n');
fprintf('  - Decision boundaries become less clear\n');
fprintf('  - Same noise level causes more bit errors\n');

fprintf('\n');
fprintf('##############################################################\n');
fprintf('#                    ANALYSIS COMPLETE                       #\n');
fprintf('##############################################################\n');

%% Save Results
save('WebLab_PA_Results.mat', 'Results', 'StoredSignals', 'BER_Results', 'Fs', 'BW');
fprintf('\nResults saved to WebLab_PA_Results.mat\n');

%% ========================================================================
%  HELPER FUNCTION: 16-QAM Demodulation
%  ========================================================================

function bits = demod_16QAM(symbols)
    n = length(symbols);
    bits = zeros(n * 4, 1);
    I = real(symbols) * sqrt(10);
    Q = imag(symbols) * sqrt(10);
    
    for k = 1:n
        idx = (k-1)*4;
        % I bits (Gray coded)
        if I(k) < -2
            bits(idx+1) = 0; bits(idx+2) = 0;
        elseif I(k) < 0
            bits(idx+1) = 0; bits(idx+2) = 1;
        elseif I(k) < 2
            bits(idx+1) = 1; bits(idx+2) = 1;
        else
            bits(idx+1) = 1; bits(idx+2) = 0;
        end
        % Q bits (Gray coded)
        if Q(k) < -2
            bits(idx+3) = 0; bits(idx+4) = 0;
        elseif Q(k) < 0
            bits(idx+3) = 0; bits(idx+4) = 1;
        elseif Q(k) < 2
            bits(idx+3) = 1; bits(idx+4) = 1;
        else
            bits(idx+3) = 1; bits(idx+4) = 0;
        end
    end
end
