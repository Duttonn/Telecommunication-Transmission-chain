%% PA Analysis - Offline Simulation Version
% This script performs comprehensive PA analysis using a simulated PA model
% Use this for testing without WebLab connection
%
% Features:
% - Signal quality metrics (EVM, ACPR, power, efficiency)
% - QPSK and 16-QAM modulated signal transmission
% - Constellation degradation visualization
% - Spectral regrowth analysis
% - BER comparison with and without PA distortion
%
% Date: November 2025

clear all; close all; clc;

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

ModulationType = '16QAM';  % 'QPSK' or '16QAM'
NumSymbols = 20000;
SamplesPerSymbol = 8;
Fs = 200e6;  % Sampling frequency
SymbolRate = Fs / SamplesPerSymbol;
RolloffFactor = 0.25;

% PA Model Parameters (Rapp Model for solid-state PA)
PA.Gain_dB = 40;           % Small-signal gain in dB
PA.Psat_dBm = 30;          % Output saturation power in dBm
PA.p = 2;                  % Smoothness factor (higher = sharper compression)
PA.Vdc = 28;               % Supply voltage (V)
PA.Idc_quiescent = 0.5;    % Quiescent current (A)

% Input power for detailed analysis
RMSin_dBm = -15;  % Input power in dBm

% ACPR parameters
BW = SymbolRate * (1 + RolloffFactor);
ACPR_params.BW = BW;
ACPR_params.Offset = 1.5 * BW;

fprintf('==========================================\n');
fprintf('   PA ANALYSIS - SIMULATION MODE\n');
fprintf('==========================================\n');
fprintf('Modulation: %s\n', ModulationType);
fprintf('Symbols: %d\n', NumSymbols);
fprintf('Symbol Rate: %.2f MHz\n', SymbolRate/1e6);
fprintf('Signal BW: %.2f MHz\n', BW/1e6);
fprintf('Input Power: %.1f dBm\n', RMSin_dBm);
fprintf('==========================================\n\n');

%% ========================================================================
%  GENERATE MODULATED SIGNAL
%  ========================================================================

fprintf('Generating %s signal...\n', ModulationType);

% Modulation parameters
switch ModulationType
    case 'QPSK'
        BitsPerSymbol = 2;
        M = 4;
    case '16QAM'
        BitsPerSymbol = 4;
        M = 16;
end

% Generate random bits
NumBits = NumSymbols * BitsPerSymbol;
txBits = randi([0 1], NumBits, 1);

% Modulate
txSymbols = modulate_signal(txBits, ModulationType);

% Pulse shaping with RRC filter
FilterSpan = 10;
RRCFilter = rcosdesign(RolloffFactor, FilterSpan, SamplesPerSymbol, 'sqrt');

% Upsample and filter
txSignal_up = upsample(txSymbols, SamplesPerSymbol);
txSignal = filter(RRCFilter, 1, txSignal_up);

% Remove filter delay
FilterDelay = FilterSpan * SamplesPerSymbol / 2;
txSignal = txSignal(FilterDelay+1:end);
txSignal = txSignal(1:NumSymbols*SamplesPerSymbol);

% Set input power level
txSignal = set_power_dbm(txSignal, RMSin_dBm);

% Calculate input metrics
Pin_dBm = calculate_power_dbm(txSignal);
PAPR_in = calculate_papr(txSignal);

fprintf('Input Power: %.2f dBm\n', Pin_dBm);
fprintf('Input PAPR: %.2f dB\n', PAPR_in);

%% ========================================================================
%  SIMULATE PA (Rapp Model with Memory)
%  ========================================================================

fprintf('\nSimulating PA...\n');

% Apply PA model
[PAout, PA_metrics] = simulate_PA(txSignal, PA);

% Calculate output metrics
Pout_dBm = calculate_power_dbm(PAout);
PAPR_out = calculate_papr(PAout);

fprintf('Output Power: %.2f dBm\n', Pout_dBm);
fprintf('Output PAPR: %.2f dB\n', PAPR_out);
fprintf('Measured Gain: %.2f dB\n', Pout_dBm - Pin_dBm);

%% ========================================================================
%  MEASURE SIGNAL QUALITY METRICS
%  ========================================================================

fprintf('\n==========================================\n');
fprintf('   SIGNAL QUALITY METRICS\n');
fprintf('==========================================\n');

% --- EVM Calculation ---
% Normalize PA output for EVM calculation
PAout_norm = PAout * (rms(txSignal) / rms(PAout));
error_signal = PAout_norm - txSignal;
EVM_percent = 100 * rms(error_signal) / rms(txSignal);
EVM_dB = 20 * log10(EVM_percent / 100);

fprintf('\n--- EVM (Error Vector Magnitude) ---\n');
fprintf('EVM: %.2f%% (%.2f dB)\n', EVM_percent, EVM_dB);

% --- ACPR Calculation ---
[ACPR_in, PSD_in] = calculate_acpr(txSignal, Fs, ACPR_params);
[ACPR_out, PSD_out] = calculate_acpr(PAout, Fs, ACPR_params);

fprintf('\n--- ACPR (Adjacent Channel Power Ratio) ---\n');
fprintf('Input ACPR  (L1/U1): %.2f / %.2f dB\n', ACPR_in.L1, ACPR_in.U1);
fprintf('Output ACPR (L1/U1): %.2f / %.2f dB\n', ACPR_out.L1, ACPR_out.U1);
fprintf('ACPR Degradation: %.2f dB\n', ACPR_out.L1 - ACPR_in.L1);

% --- Power Measurements ---
fprintf('\n--- Power Measurements ---\n');
fprintf('Input Power (Pin):   %.2f dBm\n', Pin_dBm);
fprintf('Output Power (Pout): %.2f dBm\n', Pout_dBm);
fprintf('Power Gain:          %.2f dB\n', Pout_dBm - Pin_dBm);

% --- DC Supply ---
Idc = PA.Idc_quiescent + 0.5 * (10^((Pout_dBm-30)/10)) / PA.Vdc;
Pdc = PA.Vdc * Idc;

fprintf('\n--- Supply Voltage & Current ---\n');
fprintf('Supply Voltage (Vdc): %.1f V\n', PA.Vdc);
fprintf('Supply Current (Idc): %.3f A\n', Idc);
fprintf('DC Power (Pdc):       %.2f W\n', Pdc);

% --- Power Efficiency ---
Pout_W = 10^((Pout_dBm - 30) / 10);
Pin_W = 10^((Pin_dBm - 30) / 10);
Drain_Efficiency = 100 * Pout_W / Pdc;
PAE = 100 * (Pout_W - Pin_W) / Pdc;

fprintf('\n--- Power Efficiency ---\n');
fprintf('Drain Efficiency: %.2f%%\n', Drain_Efficiency);
fprintf('Power Added Efficiency (PAE): %.2f%%\n', PAE);

%% ========================================================================
%  VISUALIZATION 1: CONSTELLATION DEGRADATION
%  ========================================================================

fprintf('\nGenerating constellation plots...\n');

figure('Name', 'Constellation Degradation', 'Position', [50, 50, 1400, 450]);

% Ideal constellation
subplot(1,3,1);
plot_constellation(txSymbols, ModulationType);
title('Ideal Constellation (No PA)', 'FontSize', 12);

% Apply matched filter to PA output
rxSignal = filter(RRCFilter, 1, [PAout; zeros(FilterDelay,1)]);
rxSignal = rxSignal(FilterDelay+1:end);
rxSymbols_PA = rxSignal(1:SamplesPerSymbol:NumSymbols*SamplesPerSymbol);

% Normalize received symbols
scale_factor = mean(abs(txSymbols)) / mean(abs(rxSymbols_PA));
rxSymbols_PA = rxSymbols_PA * scale_factor;

subplot(1,3,2);
plot_constellation(rxSymbols_PA, ModulationType);
title(sprintf('After PA (Pin=%.0fdBm, EVM=%.1f%%)', Pin_dBm, EVM_percent), 'FontSize', 12);

% Heavily compressed (simulate higher input power)
txSignal_high = set_power_dbm(txSignal, Pin_dBm + 8);
[PAout_high, ~] = simulate_PA(txSignal_high, PA);
rxSignal_high = filter(RRCFilter, 1, [PAout_high; zeros(FilterDelay,1)]);
rxSignal_high = rxSignal_high(FilterDelay+1:end);
rxSymbols_high = rxSignal_high(1:SamplesPerSymbol:NumSymbols*SamplesPerSymbol);
rxSymbols_high = rxSymbols_high * (mean(abs(txSymbols)) / mean(abs(rxSymbols_high)));

subplot(1,3,3);
plot_constellation(rxSymbols_high, ModulationType);
title(sprintf('After PA (Pin=%.0fdBm) - Compressed', Pin_dBm + 8), 'FontSize', 12);

sgtitle('Constellation Degradation due to PA Nonlinearity', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  VISUALIZATION 2: SPECTRAL REGROWTH
%  ========================================================================

fprintf('Generating spectral regrowth plot...\n');

figure('Name', 'Spectral Regrowth', 'Position', [50, 100, 1000, 600]);

% Calculate PSDs
nfft = 2048;
[Pxx_in, f] = pwelch(txSignal, blackman(nfft), nfft/2, nfft, Fs, 'centered');
[Pxx_out, ~] = pwelch(PAout, blackman(nfft), nfft/2, nfft, Fs, 'centered');

% Normalize and convert to dB
Pxx_in_dB = 10*log10(Pxx_in / max(Pxx_in));
Pxx_out_dB = 10*log10(Pxx_out / max(Pxx_out));

% Plot
plot(f/1e6, Pxx_in_dB, 'b', 'LineWidth', 2, 'DisplayName', 'PA Input');
hold on;
plot(f/1e6, Pxx_out_dB, 'r', 'LineWidth', 2, 'DisplayName', 'PA Output');

% Mark channel boundaries
xline(-BW/2e6, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(BW/2e6, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% Mark adjacent channels
xline(-ACPR_params.Offset/1e6 - BW/2e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(-ACPR_params.Offset/1e6 + BW/2e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(ACPR_params.Offset/1e6 - BW/2e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(ACPR_params.Offset/1e6 + BW/2e6, ':m', 'LineWidth', 1, 'HandleVisibility', 'off');

hold off;

xlabel('Frequency (MHz)', 'FontSize', 12);
ylabel('Normalized PSD (dB)', 'FontSize', 12);
title('Spectral Regrowth due to PA Nonlinearity', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 11);
grid on;
ylim([-80 5]);
xlim([-Fs/2e6 Fs/2e6]);

% Add annotations
annotation('textbox', [0.15, 0.75, 0.2, 0.1], 'String', ...
    sprintf('Signal BW: %.1f MHz\nACPR Degradation: %.1f dB', BW/1e6, ACPR_out.L1 - ACPR_in.L1), ...
    'EdgeColor', 'none', 'FontSize', 10, 'BackgroundColor', [1 1 1 0.7]);

%% ========================================================================
%  VISUALIZATION 3: BER COMPARISON
%  ========================================================================

fprintf('Computing BER curves...\n');

% SNR range
SNR_dB = 0:2:30;
numSNR = length(SNR_dB);

% Number of symbols for BER test
NumBER_Symbols = 50000;
BER_bits = randi([0 1], NumBER_Symbols * BitsPerSymbol, 1);
BER_symbols = modulate_signal(BER_bits, ModulationType);

% Initialize BER arrays
BER_ideal = zeros(1, numSNR);
BER_withPA = zeros(1, numSNR);

for idx = 1:numSNR
    snr = SNR_dB(idx);
    
    % Ideal channel (no PA)
    rx_ideal = awgn(BER_symbols, snr, 'measured');
    bits_ideal = demodulate_signal(rx_ideal, ModulationType);
    BER_ideal(idx) = sum(BER_bits ~= bits_ideal) / length(BER_bits);
    
    % With PA distortion (using Rapp model)
    symbols_pa = apply_pa_to_symbols(BER_symbols, PA);
    rx_pa = awgn(symbols_pa, snr, 'measured');
    bits_pa = demodulate_signal(rx_pa, ModulationType);
    BER_withPA(idx) = sum(BER_bits ~= bits_pa) / length(BER_bits);
end

% Theoretical BER
switch ModulationType
    case 'QPSK'
        BER_theory = 0.5 * erfc(sqrt(10.^(SNR_dB/10)));
    case '16QAM'
        BER_theory = (3/8) * erfc(sqrt((2/5) * 10.^(SNR_dB/10)));
end

% Plot BER
figure('Name', 'BER Comparison', 'Position', [50, 150, 900, 600]);

semilogy(SNR_dB, BER_theory, 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
hold on;
semilogy(SNR_dB, BER_ideal, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'DisplayName', 'AWGN Only (No PA)');
semilogy(SNR_dB, BER_withPA, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, ...
    'DisplayName', 'With PA Distortion');
hold off;

xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);
title(sprintf('BER Performance Comparison - %s', ModulationType), 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 11);
grid on;
ylim([1e-6 1]);
xlim([0 30]);

% Calculate SNR penalty at BER = 1e-3
target_BER = 1e-3;
if any(BER_ideal < target_BER) && any(BER_withPA < target_BER)
    snr_ideal = interp1(log10(BER_ideal(BER_ideal>0)), SNR_dB(BER_ideal>0), log10(target_BER));
    snr_pa = interp1(log10(BER_withPA(BER_withPA>0)), SNR_dB(BER_withPA>0), log10(target_BER));
    snr_penalty = snr_pa - snr_ideal;
    
    annotation('textbox', [0.55, 0.6, 0.3, 0.15], 'String', ...
        sprintf('SNR Penalty at BER=10^{-3}:\n%.2f dB', snr_penalty), ...
        'EdgeColor', 'k', 'FontSize', 11, 'BackgroundColor', [1 1 1 0.8]);
end

%% ========================================================================
%  VISUALIZATION 4: PA CHARACTERISTICS
%  ========================================================================

fprintf('Generating PA characteristic plots...\n');

figure('Name', 'PA Characteristics', 'Position', [50, 200, 1200, 400]);

% Generate AM/AM and AM/PM data
input_amp = linspace(0, 2, 1000);
[output_amp, phase_shift] = pa_characteristics(input_amp, PA);

% AM/AM characteristic
subplot(1,3,1);
plot(input_amp, output_amp, 'b', 'LineWidth', 2);
hold on;
plot(input_amp, input_amp * 10^(PA.Gain_dB/20), 'k--', 'LineWidth', 1.5);
hold off;
xlabel('Input Amplitude (normalized)');
ylabel('Output Amplitude (normalized)');
title('AM/AM Characteristic');
legend('Actual', 'Linear', 'Location', 'southeast');
grid on;

% AM/PM characteristic
subplot(1,3,2);
plot(input_amp, phase_shift * 180/pi, 'r', 'LineWidth', 2);
xlabel('Input Amplitude (normalized)');
ylabel('Phase Shift (degrees)');
title('AM/PM Characteristic');
grid on;

% Gain compression
subplot(1,3,3);
gain = 20*log10(output_amp ./ input_amp);
gain(1) = gain(2);  % Fix division by zero
plot(20*log10(input_amp + 1e-10), gain, 'g', 'LineWidth', 2);
xlabel('Input Power (dB, normalized)');
ylabel('Gain (dB)');
title('Gain Compression');
grid on;
ylim([PA.Gain_dB - 10, PA.Gain_dB + 2]);

sgtitle('Power Amplifier Characteristics', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  SUMMARY TABLE
%  ========================================================================

fprintf('\n==========================================\n');
fprintf('   SUMMARY\n');
fprintf('==========================================\n');
fprintf('| Metric                  | Value        |\n');
fprintf('|-------------------------|-------------|\n');
fprintf('| Modulation              | %-12s|\n', ModulationType);
fprintf('| Input Power             | %7.2f dBm |\n', Pin_dBm);
fprintf('| Output Power            | %7.2f dBm |\n', Pout_dBm);
fprintf('| Gain                    | %7.2f dB  |\n', Pout_dBm - Pin_dBm);
fprintf('| EVM                     | %7.2f %%   |\n', EVM_percent);
fprintf('| ACPR (Lower)            | %7.2f dB  |\n', ACPR_out.L1);
fprintf('| ACPR (Upper)            | %7.2f dB  |\n', ACPR_out.U1);
fprintf('| Supply Voltage          | %7.1f V   |\n', PA.Vdc);
fprintf('| Supply Current          | %7.3f A   |\n', Idc);
fprintf('| DC Power                | %7.2f W   |\n', Pdc);
fprintf('| Drain Efficiency        | %7.2f %%   |\n', Drain_Efficiency);
fprintf('| PAE                     | %7.2f %%   |\n', PAE);
fprintf('==========================================\n');

fprintf('\nAnalysis complete! All figures generated.\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function symbols = modulate_signal(bits, modType)
    switch modType
        case 'QPSK'
            bitsPerSym = 2;
            numSym = length(bits) / bitsPerSym;
            bits_reshape = reshape(bits, bitsPerSym, numSym).';
            symbols = (1/sqrt(2)) * ((2*bits_reshape(:,1)-1) + 1j*(2*bits_reshape(:,2)-1));
        case '16QAM'
            bitsPerSym = 4;
            numSym = length(bits) / bitsPerSym;
            bits_reshape = reshape(bits, bitsPerSym, numSym).';
            I = 2*bits_reshape(:,1) + bits_reshape(:,2);
            Q = 2*bits_reshape(:,3) + bits_reshape(:,4);
            I_map = [-3 -1 3 1];
            Q_map = [-3 -1 3 1];
            symbols = (1/sqrt(10)) * (I_map(I+1).' + 1j*Q_map(Q+1).');
    end
end

function bits = demodulate_signal(symbols, modType)
    numSym = length(symbols);
    switch modType
        case 'QPSK'
            bits = zeros(numSym * 2, 1);
            bits(1:2:end) = real(symbols) > 0;
            bits(2:2:end) = imag(symbols) > 0;
        case '16QAM'
            bits = zeros(numSym * 4, 1);
            I = real(symbols) * sqrt(10);
            Q = imag(symbols) * sqrt(10);
            for k = 1:numSym
                idx = (k-1)*4;
                if I(k) < -2, bits(idx+1:idx+2) = [0;0];
                elseif I(k) < 0, bits(idx+1:idx+2) = [0;1];
                elseif I(k) < 2, bits(idx+1:idx+2) = [1;1];
                else, bits(idx+1:idx+2) = [1;0]; end
                if Q(k) < -2, bits(idx+3:idx+4) = [0;0];
                elseif Q(k) < 0, bits(idx+3:idx+4) = [0;1];
                elseif Q(k) < 2, bits(idx+3:idx+4) = [1;1];
                else, bits(idx+3:idx+4) = [1;0]; end
            end
    end
end

function signal_out = set_power_dbm(signal_in, power_dbm)
    current_power = 10*log10(mean(abs(signal_in).^2) / 50 * 1000);
    scale = 10^((power_dbm - current_power) / 20);
    signal_out = signal_in * scale;
end

function power_dbm = calculate_power_dbm(signal)
    power_dbm = 10*log10(mean(abs(signal).^2) / 50 * 1000);
end

function papr = calculate_papr(signal)
    papr = 10*log10(max(abs(signal).^2) / mean(abs(signal).^2));
end

function [y, metrics] = simulate_PA(x, PA)
    % Rapp model for solid-state PA
    A_sat = sqrt(2 * 50 * 10^((PA.Psat_dBm - 30)/10));  % Saturation amplitude
    G = 10^(PA.Gain_dB / 20);  % Linear gain
    p = PA.p;  % Smoothness
    
    % Apply gain
    x_amp = G * x;
    
    % Apply AM/AM (Rapp model)
    abs_x = abs(x_amp);
    compression = 1 ./ (1 + (abs_x / A_sat).^(2*p)).^(1/(2*p));
    
    % Apply AM/PM (simplified model)
    phase_shift = 0.1 * (abs_x / A_sat).^2;  % Quadratic phase distortion
    
    y = x_amp .* compression .* exp(1j * phase_shift);
    
    metrics.compression = mean(compression);
    metrics.max_phase_shift = max(phase_shift) * 180/pi;
end

function symbols_out = apply_pa_to_symbols(symbols_in, PA)
    A_sat = 1.5;  % Normalized saturation
    p = PA.p;
    abs_sym = abs(symbols_in);
    compression = 1 ./ (1 + (abs_sym / A_sat).^(2*p)).^(1/(2*p));
    phase_shift = 0.05 * (abs_sym / A_sat).^2;
    symbols_out = symbols_in .* compression .* exp(1j * phase_shift);
end

function [output_amp, phase_shift] = pa_characteristics(input_amp, PA)
    A_sat = 1.0;
    p = PA.p;
    G = 10^(PA.Gain_dB / 20);
    
    x_amp = G * input_amp;
    compression = 1 ./ (1 + (x_amp / (A_sat * G)).^(2*p)).^(1/(2*p));
    output_amp = x_amp .* compression;
    phase_shift = 0.1 * (input_amp / A_sat).^2;
end

function [ACPR, PSD] = calculate_acpr(signal, Fs, params)
    BW = params.BW;
    Offset = params.Offset;
    
    nfft = 2048;
    [Pxx, f] = pwelch(signal, blackman(nfft), nfft/2, nfft, Fs, 'centered');
    
    PSD.Data = Pxx;
    PSD.Frequencies = f;
    
    % Main channel power
    main_idx = (f >= -BW/2) & (f <= BW/2);
    P_main = sum(Pxx(main_idx));
    
    % Lower adjacent channel
    lower_idx = (f >= -Offset-BW/2) & (f <= -Offset+BW/2);
    P_lower = sum(Pxx(lower_idx));
    
    % Upper adjacent channel
    upper_idx = (f >= Offset-BW/2) & (f <= Offset+BW/2);
    P_upper = sum(Pxx(upper_idx));
    
    ACPR.L1 = 10*log10(P_lower / P_main);
    ACPR.U1 = 10*log10(P_upper / P_main);
    ACPR.BW = BW;
    ACPR.Offset = Offset;
end

function plot_constellation(symbols, modType)
    plot(real(symbols), imag(symbols), '.', 'MarkerSize', 1, 'Color', [0 0.4 0.8]);
    hold on;
    
    % Reference constellation
    switch modType
        case 'QPSK'
            ref = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];
        case '16QAM'
            [I, Q] = meshgrid([-3 -1 1 3], [-3 -1 1 3]);
            ref = (1/sqrt(10)) * (I(:) + 1j*Q(:));
    end
    plot(real(ref), imag(ref), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    hold off;
    
    xlabel('In-Phase', 'FontSize', 10);
    ylabel('Quadrature', 'FontSize', 10);
    axis equal; grid on;
    maxVal = max(abs([real(symbols); imag(symbols)])) * 1.3;
    maxVal = max(maxVal, 1.5);
    xlim([-maxVal maxVal]);
    ylim([-maxVal maxVal]);
end
