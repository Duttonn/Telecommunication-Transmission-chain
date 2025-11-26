%% QPSK vs 16-QAM Comparison under PA Distortion
% This script compares QPSK and 16-QAM performance under PA nonlinearity
%
% Date: November 2025

clear all; close all; clc;

%% Configuration
NumSymbols = 30000;
SamplesPerSymbol = 8;
Fs = 200e6;
RolloffFactor = 0.25;

% PA parameters
PA.Gain_dB = 40;
PA.Psat_dBm = 30;
PA.p = 2;
PA.Vdc = 28;

% Input power levels to test
Pin_levels = [-20, -15, -12, -10];

fprintf('==============================================\n');
fprintf('   QPSK vs 16-QAM Comparison under PA\n');
fprintf('==============================================\n\n');

%% Run analysis for both modulations

modTypes = {'QPSK', 'QAM16'};
results = struct();

for modIdx = 1:2
    modType = modTypes{modIdx};
    fprintf('Analyzing %s...\n', modType);
    
    % Get bits per symbol
    if strcmp(modType, 'QPSK')
        bitsPerSym = 2;
    else  % QAM16
        bitsPerSym = 4;
    end
    
    % Generate signal
    bits = randi([0 1], NumSymbols * bitsPerSym, 1);
    symbols = modulate(bits, modType);
    
    % Pulse shaping
    FilterSpan = 10;
    RRCFilter = rcosdesign(RolloffFactor, FilterSpan, SamplesPerSymbol, 'sqrt');
    signal_up = upsample(symbols, SamplesPerSymbol);
    signal = filter(RRCFilter, 1, signal_up);
    FilterDelay = FilterSpan * SamplesPerSymbol / 2;
    
    % Trim to valid length (account for filter delay)
    validLength = length(signal) - FilterDelay;
    signal = signal(FilterDelay+1:FilterDelay+validLength);
    signal = signal(1:min(length(signal), NumSymbols*SamplesPerSymbol));
    
    % Store results for each power level
    for pIdx = 1:length(Pin_levels)
        Pin = Pin_levels(pIdx);
        
        % Set power
        signal_pwr = setPower(signal, Pin);
        
        % Simulate PA
        [pa_out, ~] = simulatePA(signal_pwr, PA);
        
        % Normalize
        pa_out_norm = pa_out * (rms(signal_pwr) / rms(pa_out));
        
        % Calculate EVM
        evm = 100 * rms(pa_out_norm - signal_pwr) / rms(signal_pwr);
        
        % Calculate ACPR
        BW = (Fs/SamplesPerSymbol) * (1 + RolloffFactor);
        acpr_val = calcACPR(pa_out, Fs, BW);
        
        % Store
        results.(modType).Pin(pIdx) = Pin;
        results.(modType).EVM(pIdx) = evm;
        results.(modType).ACPR(pIdx) = acpr_val;
    end
    
    % BER analysis
    SNR_dB = 0:2:30;
    [ber_ideal, ber_pa] = calcBER(modType, SNR_dB, PA);
    results.(modType).SNR = SNR_dB;
    results.(modType).BER_ideal = ber_ideal;
    results.(modType).BER_PA = ber_pa;
end

%% Plot Comparison

figure('Name', 'QPSK vs 16-QAM Comparison', 'Position', [50, 50, 1400, 900]);

% EVM vs Input Power
subplot(2,3,1);
plot(Pin_levels, results.QPSK.EVM, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(Pin_levels, results.QAM16.EVM, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
hold off;
xlabel('Input Power (dBm)');
ylabel('EVM (%)');
title('EVM vs Input Power');
legend('QPSK', '16-QAM', 'Location', 'northwest');
grid on;

% ACPR vs Input Power
subplot(2,3,2);
plot(Pin_levels, results.QPSK.ACPR, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(Pin_levels, results.QAM16.ACPR, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
hold off;
xlabel('Input Power (dBm)');
ylabel('ACPR (dB)');
title('ACPR vs Input Power');
legend('QPSK', '16-QAM', 'Location', 'southwest');
grid on;

% BER Comparison
subplot(2,3,3);
semilogy(results.QPSK.SNR, results.QPSK.BER_ideal, 'b--', 'LineWidth', 1.5);
hold on;
semilogy(results.QPSK.SNR, results.QPSK.BER_PA, 'b-o', 'LineWidth', 2);
semilogy(results.QAM16.SNR, results.QAM16.BER_ideal, 'r--', 'LineWidth', 1.5);
semilogy(results.QAM16.SNR, results.QAM16.BER_PA, 'r-s', 'LineWidth', 2);
hold off;
xlabel('SNR (dB)');
ylabel('BER');
title('BER Comparison');
legend('QPSK Ideal', 'QPSK + PA', '16-QAM Ideal', '16-QAM + PA', 'Location', 'southwest');
grid on;
ylim([1e-6 1]);

% Constellation - QPSK
subplot(2,3,4);
bits_q = randi([0 1], 5000*2, 1);
sym_q = modulate(bits_q, 'QPSK');
sym_q_pa = applyPA(sym_q, PA);
plot(real(sym_q_pa), imag(sym_q_pa), '.', 'MarkerSize', 2, 'Color', [0 0.4 0.8]);
hold on;
ref_q = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];
plot(real(ref_q), imag(ref_q), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
hold off;
xlabel('I'); ylabel('Q');
title('QPSK Constellation (with PA)');
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);

% Constellation - 16QAM
subplot(2,3,5);
bits_16 = randi([0 1], 5000*4, 1);
sym_16 = modulate(bits_16, 'QAM16');
sym_16_pa = applyPA(sym_16, PA);
plot(real(sym_16_pa), imag(sym_16_pa), '.', 'MarkerSize', 2, 'Color', [0.8 0.2 0]);
hold on;
[I, Q] = meshgrid([-3 -1 1 3], [-3 -1 1 3]);
ref_16 = (1/sqrt(10)) * (I(:) + 1j*Q(:));
plot(real(ref_16), imag(ref_16), 'ko', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
xlabel('I'); ylabel('Q');
title('16-QAM Constellation (with PA)');
axis equal; grid on; xlim([-2 2]); ylim([-2 2]);

% Summary Table
subplot(2,3,6);
axis off;
text(0.1, 0.95, 'Performance Summary', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.80, sprintf('At Pin = %.0f dBm:', Pin_levels(3)), 'FontSize', 12);
text(0.1, 0.65, sprintf('QPSK EVM: %.2f%%', results.QPSK.EVM(3)), 'FontSize', 11);
text(0.1, 0.55, sprintf('16-QAM EVM: %.2f%%', results.QAM16.EVM(3)), 'FontSize', 11);
text(0.1, 0.40, sprintf('QPSK ACPR: %.1f dB', results.QPSK.ACPR(3)), 'FontSize', 11);
text(0.1, 0.30, sprintf('16-QAM ACPR: %.1f dB', results.QAM16.ACPR(3)), 'FontSize', 11);
text(0.1, 0.15, 'Note: 16-QAM is more sensitive', 'FontSize', 10, 'FontAngle', 'italic');
text(0.1, 0.05, 'to PA nonlinearity due to', 'FontSize', 10, 'FontAngle', 'italic');
text(0.1, -0.05, 'smaller constellation spacing', 'FontSize', 10, 'FontAngle', 'italic');

sgtitle('QPSK vs 16-QAM Performance Comparison under PA Nonlinearity', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%  DETAILED REPORT - ANSWERS TO KEY QUESTIONS
%  ========================================================================

fprintf('\n');
fprintf('##############################################################\n');
fprintf('#                  DETAILED ANALYSIS REPORT                  #\n');
fprintf('##############################################################\n');

%% --- Q1: What is EVM and what do the results show? ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q1: EVM (Error Vector Magnitude) Analysis\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('DEFINITION:\n');
fprintf('  EVM measures the difference between the ideal transmitted symbol\n');
fprintf('  and the actual received symbol. It quantifies signal distortion.\n');
fprintf('  \n');
fprintf('  Formula: EVM(%%) = (RMS error / RMS reference) x 100\n');
fprintf('\n');
fprintf('RESULTS:\n');
for pIdx = 1:length(Pin_levels)
    fprintf('  At Pin = %3d dBm:  QPSK EVM = %5.2f%%,  16-QAM EVM = %5.2f%%\n', ...
        Pin_levels(pIdx), results.QPSK.EVM(pIdx), results.QAM16.EVM(pIdx));
end
fprintf('\n');
fprintf('INTERPRETATION:\n');
fprintf('  - EVM increases with input power (more compression = more distortion)\n');
fprintf('  - 16-QAM has similar EVM values but is MORE SENSITIVE to it\n');
fprintf('  - Why? 16-QAM symbols are closer together, so same EVM causes more errors\n');
fprintf('  - Typical limits: QPSK < 17.5%%, 16-QAM < 12.5%% (3GPP LTE)\n');

%% --- Q2: What is ACPR and what causes spectral regrowth? ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q2: ACPR (Adjacent Channel Power Ratio) & Spectral Regrowth\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('DEFINITION:\n');
fprintf('  ACPR = Power in adjacent channel / Power in main channel (in dB)\n');
fprintf('  It measures how much signal "leaks" into neighboring frequency bands.\n');
fprintf('\n');
fprintf('RESULTS:\n');
for pIdx = 1:length(Pin_levels)
    fprintf('  At Pin = %3d dBm:  QPSK ACPR = %5.1f dB,  16-QAM ACPR = %5.1f dB\n', ...
        Pin_levels(pIdx), results.QPSK.ACPR(pIdx), results.QAM16.ACPR(pIdx));
end
fprintf('\n');
fprintf('WHAT CAUSES SPECTRAL REGROWTH?\n');
fprintf('  1. PA nonlinearity (AM/AM compression) generates harmonics\n');
fprintf('  2. These harmonics create intermodulation products\n');
fprintf('  3. Intermodulation spreads signal energy outside the channel\n');
fprintf('  4. Higher input power = more compression = worse ACPR\n');
fprintf('\n');
fprintf('INTERPRETATION:\n');
fprintf('  - More negative ACPR is BETTER (less leakage)\n');
fprintf('  - ACPR degrades (becomes less negative) as PA compresses\n');
fprintf('  - Typical requirement: ACPR < -45 dB for LTE\n');

%% --- Q3: How does PA affect BER? ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q3: BER Performance - With vs Without PA Distortion\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('DEFINITION:\n');
fprintf('  BER = Number of bit errors / Total bits transmitted\n');
fprintf('\n');

% Calculate SNR penalty at BER = 1e-3
target_ber = 1e-3;
fprintf('SNR PENALTY AT BER = 10^-3:\n');

for modIdx = 1:2
    modType = modTypes{modIdx};
    ber_ideal = results.(modType).BER_ideal;
    ber_pa = results.(modType).BER_PA;
    snr = results.(modType).SNR;
    
    % Find SNR for target BER (interpolate)
    valid_ideal = ber_ideal > 0 & ber_ideal < 1;
    valid_pa = ber_pa > 0 & ber_pa < 1;
    
    if sum(valid_ideal) > 2 && sum(valid_pa) > 2
        try
            snr_ideal = interp1(log10(ber_ideal(valid_ideal)), snr(valid_ideal), log10(target_ber), 'linear', 'extrap');
            snr_pa = interp1(log10(ber_pa(valid_pa)), snr(valid_pa), log10(target_ber), 'linear', 'extrap');
            penalty = snr_pa - snr_ideal;
            fprintf('  %s: SNR penalty = %.2f dB\n', modType, penalty);
            fprintf('       (Need %.1f dB SNR ideal vs %.1f dB with PA)\n', snr_ideal, snr_pa);
        catch
            fprintf('  %s: Could not calculate penalty (BER too high/low)\n', modType);
        end
    end
end

fprintf('\n');
fprintf('WHY DOES PA DEGRADE BER?\n');
fprintf('  1. PA compression distorts symbol positions (constellation warping)\n');
fprintf('  2. AM/PM conversion adds phase noise\n');
fprintf('  3. Symbols move closer to decision boundaries\n');
fprintf('  4. Same noise level now causes more decision errors\n');
fprintf('\n');
fprintf('KEY OBSERVATION:\n');
fprintf('  - 16-QAM suffers MORE from PA distortion than QPSK\n');
fprintf('  - Reason: 16-QAM has 4 bits/symbol vs 2 bits/symbol\n');
fprintf('  - Constellation points are closer together in 16-QAM\n');
fprintf('  - Same amount of distortion causes more bit errors\n');

%% --- Q4: Power Efficiency ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q4: Power Efficiency\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('DEFINITIONS:\n');
fprintf('  Drain Efficiency = Pout_RF / Pdc x 100%%\n');
fprintf('  PAE (Power Added Efficiency) = (Pout - Pin) / Pdc x 100%%\n');
fprintf('\n');
fprintf('PA PARAMETERS USED:\n');
fprintf('  Supply Voltage (Vdc): %.1f V\n', PA.Vdc);
fprintf('  Saturation Power: %.1f dBm\n', PA.Psat_dBm);
fprintf('  Small-signal Gain: %.1f dB\n', PA.Gain_dB);
fprintf('\n');
fprintf('EFFICIENCY vs LINEARITY TRADEOFF:\n');
fprintf('  - High efficiency requires operating near saturation\n');
fprintf('  - But saturation causes nonlinear distortion!\n');
fprintf('  - Engineers must balance: more efficiency = more distortion\n');
fprintf('  - Solution: Use DPD (Digital Pre-Distortion) to linearize PA\n');

%% --- Q5: Constellation Degradation ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('Q5: Constellation Degradation Explanation\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('WHAT YOU SEE IN THE CONSTELLATION PLOTS:\n');
fprintf('\n');
fprintf('  IDEAL (No PA):\n');
fprintf('    - Symbols are tight clusters at exact reference points\n');
fprintf('    - QPSK: 4 points, 16-QAM: 16 points in a square grid\n');
fprintf('\n');
fprintf('  WITH PA DISTORTION:\n');
fprintf('    - Clusters become "clouds" - symbols spread out\n');
fprintf('    - Outer symbols compress inward (AM/AM compression)\n');
fprintf('    - Phase rotation visible (AM/PM conversion)\n');
fprintf('    - 16-QAM: inner and outer symbols affected differently\n');
fprintf('\n');
fprintf('WHY OUTER SYMBOLS SUFFER MORE:\n');
fprintf('    - Outer symbols have higher amplitude\n');
fprintf('    - They hit PA compression region more\n');
fprintf('    - Inner symbols stay in linear region\n');
fprintf('    - This is called "constellation warping"\n');

%% --- Summary Table ---
fprintf('\n');
fprintf('==============================================================\n');
fprintf('SUMMARY COMPARISON TABLE\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('┌─────────────────────┬────────────────┬────────────────┐\n');
fprintf('│ Metric              │     QPSK       │    16-QAM      │\n');
fprintf('├─────────────────────┼────────────────┼────────────────┤\n');
fprintf('│ Bits per Symbol     │       2        │       4        │\n');
fprintf('│ Spectral Efficiency │    2 bps/Hz    │    4 bps/Hz    │\n');
fprintf('│ EVM (at %ddBm)      │    %5.2f %%     │    %5.2f %%     │\n', ...
    Pin_levels(3), results.QPSK.EVM(3), results.QAM16.EVM(3));
fprintf('│ ACPR (at %ddBm)     │   %5.1f dB    │   %5.1f dB    │\n', ...
    Pin_levels(3), results.QPSK.ACPR(3), results.QAM16.ACPR(3));
fprintf('│ PA Sensitivity      │     Low        │     High       │\n');
fprintf('│ Required Linearity  │     Low        │     High       │\n');
fprintf('└─────────────────────┴────────────────┴────────────────┘\n');

fprintf('\n');
fprintf('==============================================================\n');
fprintf('KEY TAKEAWAYS FOR EXAM/PRESENTATION\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('1. EVM measures constellation distortion (lower is better)\n');
fprintf('2. ACPR measures spectral leakage (more negative is better)\n');
fprintf('3. PA nonlinearity causes both EVM degradation and spectral regrowth\n');
fprintf('4. Higher-order modulation (16-QAM) is more sensitive to PA distortion\n');
fprintf('5. Tradeoff: Efficiency vs Linearity - cant have both!\n');
fprintf('6. Solution: DPD pre-distorts signal to cancel PA nonlinearity\n');
fprintf('\n');
fprintf('##############################################################\n');
fprintf('#                    END OF REPORT                           #\n');
fprintf('##############################################################\n');

%% Helper Functions

function symbols = modulate(bits, modType)
    switch modType
        case 'QPSK'
            bps = 2;
            n = length(bits)/bps;
            b = reshape(bits, bps, n).';
            symbols = (1/sqrt(2)) * ((2*b(:,1)-1) + 1j*(2*b(:,2)-1));
        case 'QAM16'
            bps = 4;
            n = length(bits)/bps;
            b = reshape(bits, bps, n).';
            I = 2*b(:,1) + b(:,2);
            Q = 2*b(:,3) + b(:,4);
            I_map = [-3 -1 3 1];
            Q_map = [-3 -1 3 1];
            symbols = (1/sqrt(10)) * (I_map(I+1).' + 1j*Q_map(Q+1).');
    end
end

function bits = demodulate(symbols, modType)
    n = length(symbols);
    switch modType
        case 'QPSK'
            bits = zeros(n*2, 1);
            bits(1:2:end) = real(symbols) > 0;
            bits(2:2:end) = imag(symbols) > 0;
        case 'QAM16'
            bits = zeros(n*4, 1);
            I = real(symbols) * sqrt(10);
            Q = imag(symbols) * sqrt(10);
            for k = 1:n
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

function s = setPower(x, pwr_dbm)
    p_curr = 10*log10(mean(abs(x).^2)/50*1000);
    s = x * 10^((pwr_dbm - p_curr)/20);
end

function [y, m] = simulatePA(x, PA)
    A_sat = sqrt(2*50*10^((PA.Psat_dBm-30)/10));
    G = 10^(PA.Gain_dB/20);
    p = PA.p;
    x_amp = G * x;
    abs_x = abs(x_amp);
    comp = 1 ./ (1 + (abs_x/A_sat).^(2*p)).^(1/(2*p));
    phase = 0.1 * (abs_x/A_sat).^2;
    y = x_amp .* comp .* exp(1j*phase);
    m.comp = mean(comp);
end

function y = applyPA(x, PA)
    A_sat = 1.5;
    p = PA.p;
    abs_x = abs(x);
    comp = 1 ./ (1 + (abs_x/A_sat).^(2*p)).^(1/(2*p));
    phase = 0.05 * (abs_x/A_sat).^2;
    y = x .* comp .* exp(1j*phase);
end

function acpr = calcACPR(signal, Fs, BW)
    nfft = 2048;
    [Pxx, f] = pwelch(signal, blackman(nfft), nfft/2, nfft, Fs, 'centered');
    Offset = 1.5 * BW;
    main_idx = (f >= -BW/2) & (f <= BW/2);
    adj_idx = (f >= Offset-BW/2) & (f <= Offset+BW/2);
    acpr = 10*log10(sum(Pxx(adj_idx)) / sum(Pxx(main_idx)));
end

function [ber_ideal, ber_pa] = calcBER(modType, SNR_dB, PA)
    if strcmp(modType, 'QPSK')
        bps = 2;
    else  % QAM16
        bps = 4;
    end
    n = 50000;
    bits = randi([0 1], n*bps, 1);
    sym = modulate(bits, modType);
    
    ber_ideal = zeros(size(SNR_dB));
    ber_pa = zeros(size(SNR_dB));
    
    for i = 1:length(SNR_dB)
        % Ideal
        rx = awgn(sym, SNR_dB(i), 'measured');
        bits_rx = demodulate(rx, modType);
        ber_ideal(i) = sum(bits ~= bits_rx) / length(bits);
        
        % With PA
        sym_pa = applyPA(sym, PA);
        rx_pa = awgn(sym_pa, SNR_dB(i), 'measured');
        bits_pa = demodulate(rx_pa, modType);
        ber_pa(i) = sum(bits ~= bits_pa) / length(bits);
    end
end
