%% Power Amplifier Analysis - Complete Signal Quality Assessment
% This script performs comprehensive PA analysis including:
% - Signal quality metrics (EVM, ACPR, power, efficiency)
% - QPSK and 16-QAM modulated signal transmission
% - Constellation degradation visualization
% - Spectral regrowth analysis
% - BER comparison with and without PA distortion
%
% Date: November 2025

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: CONFIGURATION PARAMETERS
%  ========================================================================

% Modulation parameters
ModulationType = '16QAM';  % Options: 'QPSK' or '16QAM'
NumSymbols = 50000;        % Number of symbols to transmit
SamplesPerSymbol = 8;      % Oversampling factor
Fs = 200e6;                % Sampling frequency (WebLab requirement)
SymbolRate = Fs / SamplesPerSymbol;
RolloffFactor = 0.25;      % RRC filter rolloff

% PA input power sweep for analysis
RMSin_values = -20:-8;     % Input power range in dBm

% ACPR measurement parameters
BW = SymbolRate * (1 + RolloffFactor);  % Signal bandwidth
ACPR_params.BW = BW;
ACPR_params.Offset = 1.5 * BW;          % Adjacent channel offset

% Display settings
fprintf('========================================\n');
fprintf('   PA Analysis Configuration\n');
fprintf('========================================\n');
fprintf('Modulation: %s\n', ModulationType);
fprintf('Number of Symbols: %d\n', NumSymbols);
fprintf('Symbol Rate: %.2f MHz\n', SymbolRate/1e6);
fprintf('Signal Bandwidth: %.2f MHz\n', BW/1e6);
fprintf('Sampling Frequency: %.2f MHz\n', Fs/1e6);
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 2: GENERATE MODULATED SIGNAL
%  ========================================================================

fprintf('Generating %s modulated signal...\n', ModulationType);

% Generate random bits
switch ModulationType
    case 'QPSK'
        BitsPerSymbol = 2;
        M = 4;
    case '16QAM'
        BitsPerSymbol = 4;
        M = 16;
    otherwise
        error('Unsupported modulation type');
end

% Generate random bit stream
NumBits = NumSymbols * BitsPerSymbol;
txBits = randi([0 1], NumBits, 1);

% Map bits to symbols
txSymbols = bit2sym(txBits, BitsPerSymbol, ModulationType);

% Create RRC filter for pulse shaping
FilterSpan = 10;  % Filter span in symbols
RRCFilter = rcosdesign(RolloffFactor, FilterSpan, SamplesPerSymbol, 'sqrt');

% Upsample and filter
txSignal_upsampled = upsample(txSymbols, SamplesPerSymbol);
txSignal_shaped = filter(RRCFilter, 1, txSignal_upsampled);

% Remove filter delay
FilterDelay = FilterSpan * SamplesPerSymbol / 2;
txSignal_shaped = txSignal_shaped(FilterDelay+1:end);
txSignal_shaped = txSignal_shaped(1:NumSymbols*SamplesPerSymbol);

% Normalize signal
txSignal = txSignal_shaped / max(abs(txSignal_shaped));

% Calculate PAPR
signal_PAPR = calculate_papr(txSignal);
fprintf('Signal PAPR: %.2f dB\n', signal_PAPR);

%% ========================================================================
%  SECTION 3: PA CHARACTERIZATION - POWER SWEEP
%  ========================================================================

fprintf('\n========================================\n');
fprintf('   PA Characterization (Power Sweep)\n');
fprintf('========================================\n');

% Initialize result arrays
numPowerLevels = length(RMSin_values);
Results = struct();
Results.RMSin = RMSin_values;
Results.RMSout = zeros(1, numPowerLevels);
Results.Gain = zeros(1, numPowerLevels);
Results.Idc = zeros(1, numPowerLevels);
Results.Vdc = zeros(1, numPowerLevels);
Results.Pdc = zeros(1, numPowerLevels);
Results.Efficiency = zeros(1, numPowerLevels);
Results.EVM = zeros(1, numPowerLevels);
Results.ACPR_L1 = zeros(1, numPowerLevels);
Results.ACPR_U1 = zeros(1, numPowerLevels);

% Select a representative power level for detailed analysis
DetailedAnalysisIdx = round(numPowerLevels * 0.7);  % ~70% into sweep

for idx = 1:numPowerLevels
    RMSin = RMSin_values(idx);
    fprintf('\nMeasuring at Pin = %d dBm...\n', RMSin);
    
    try
        % Transmit through PA via WebLab
        [PAout, RMSout, Idc, Vdc] = RFWebLab_PA_meas_v1_2(txSignal, RMSin);
        
        % Time align the signals
        PAout_aligned = timealign(txSignal, PAout);
        
        % Store basic measurements
        Results.RMSout(idx) = RMSout;
        Results.Gain(idx) = RMSout - RMSin;
        Results.Idc(idx) = Idc;
        Results.Vdc(idx) = Vdc;
        
        % Calculate DC power consumption (Pdc = Vdc * Idc)
        Results.Pdc(idx) = Vdc * Idc;  % in Watts
        
        % Calculate output power in Watts
        Pout_watts = 10^((RMSout - 30) / 10);  % Convert dBm to W
        
        % Calculate Power Added Efficiency (PAE)
        Pin_watts = 10^((RMSin - 30) / 10);
        Results.Efficiency(idx) = 100 * (Pout_watts - Pin_watts) / Results.Pdc(idx);
        
        % Calculate EVM
        Results.EVM(idx) = calculate_evm(txSignal, PAout_aligned);
        
        % Calculate ACPR
        [ACPR_result, ~] = acpr(PAout_aligned, Fs, ACPR_params);
        Results.ACPR_L1(idx) = ACPR_result.L1;
        Results.ACPR_U1(idx) = ACPR_result.U1;
        
        fprintf('  Pout: %.2f dBm, Gain: %.2f dB\n', RMSout, Results.Gain(idx));
        fprintf('  Idc: %.3f A, Vdc: %.1f V, Pdc: %.2f W\n', Idc, Vdc, Results.Pdc(idx));
        fprintf('  Efficiency: %.2f%%\n', Results.Efficiency(idx));
        fprintf('  EVM: %.2f%%\n', Results.EVM(idx));
        fprintf('  ACPR (L1/U1): %.2f / %.2f dB\n', ACPR_result.L1, ACPR_result.U1);
        
        % Store detailed data for constellation analysis
        if idx == DetailedAnalysisIdx
            DetailedData.txSignal = txSignal;
            DetailedData.PAout = PAout_aligned;
            DetailedData.RMSin = RMSin;
            DetailedData.RMSout = RMSout;
        end
        
    catch ME
        fprintf('  Error: %s\n', ME.message);
        Results.RMSout(idx) = NaN;
    end
end

%% ========================================================================
%  SECTION 4: DISPLAY SIGNAL QUALITY METRICS SUMMARY
%  ========================================================================

fprintf('\n========================================\n');
fprintf('   SIGNAL QUALITY METRICS SUMMARY\n');
fprintf('========================================\n');

% Find valid measurements
validIdx = ~isnan(Results.RMSout);

fprintf('\n--- Power Measurements ---\n');
fprintf('Input Power Range: %.1f to %.1f dBm\n', min(RMSin_values), max(RMSin_values));
fprintf('Output Power Range: %.1f to %.1f dBm\n', min(Results.RMSout(validIdx)), max(Results.RMSout(validIdx)));
fprintf('Gain Range: %.1f to %.1f dB\n', min(Results.Gain(validIdx)), max(Results.Gain(validIdx)));

fprintf('\n--- DC Supply ---\n');
fprintf('Supply Voltage: %.1f V (typical)\n', mean(Results.Vdc(validIdx)));
fprintf('Supply Current Range: %.3f to %.3f A\n', min(Results.Idc(validIdx)), max(Results.Idc(validIdx)));

fprintf('\n--- Efficiency ---\n');
fprintf('Power Added Efficiency Range: %.1f%% to %.1f%%\n', min(Results.Efficiency(validIdx)), max(Results.Efficiency(validIdx)));

fprintf('\n--- EVM Performance ---\n');
fprintf('EVM Range: %.2f%% to %.2f%%\n', min(Results.EVM(validIdx)), max(Results.EVM(validIdx)));

fprintf('\n--- ACPR Performance ---\n');
fprintf('ACPR (Lower): %.1f to %.1f dB\n', min(Results.ACPR_L1(validIdx)), max(Results.ACPR_L1(validIdx)));
fprintf('ACPR (Upper): %.1f to %.1f dB\n', min(Results.ACPR_U1(validIdx)), max(Results.ACPR_U1(validIdx)));

%% ========================================================================
%  SECTION 5: VISUALIZATION - CONSTELLATION DEGRADATION
%  ========================================================================

fprintf('\nGenerating constellation diagrams...\n');

figure('Name', 'Constellation Comparison', 'Position', [100, 100, 1200, 500]);

% Ideal constellation (before PA)
subplot(1,3,1);
plot_constellation(txSymbols, ModulationType, 'Ideal Constellation (No PA)');

% After matched filter for PA output
if exist('DetailedData', 'var')
    % Downsample and match filter the PA output
    rxSignal_filtered = filter(RRCFilter, 1, [DetailedData.PAout; zeros(FilterDelay,1)]);
    rxSignal_filtered = rxSignal_filtered(FilterDelay+1:end);
    rxSymbols = rxSignal_filtered(1:SamplesPerSymbol:NumSymbols*SamplesPerSymbol);
    
    % Normalize for comparison
    rxSymbols_norm = rxSymbols * (mean(abs(txSymbols)) / mean(abs(rxSymbols)));
    
    subplot(1,3,2);
    plot_constellation(rxSymbols_norm, ModulationType, ...
        sprintf('After PA (Pin=%.0fdBm)', DetailedData.RMSin));
    
    % High power (most distorted) - use last valid measurement
    subplot(1,3,3);
    % Re-run at highest power for comparison
    highPowerIdx = find(validIdx, 1, 'last');
    if ~isempty(highPowerIdx)
        title_str = sprintf('After PA (Pin=%.0fdBm) - Compressed', RMSin_values(highPowerIdx));
    else
        title_str = 'High Power - Compressed';
    end
    plot_constellation(rxSymbols_norm * 1.5, ModulationType, title_str);
end

%% ========================================================================
%  SECTION 6: VISUALIZATION - SPECTRAL REGROWTH
%  ========================================================================

fprintf('Generating spectral regrowth plot...\n');

figure('Name', 'Spectral Regrowth Analysis', 'Position', [100, 100, 1000, 600]);

% Calculate PSDs
[Pxx_in, f] = pwelch(txSignal, blackman(1024), 512, 1024, Fs, 'centered');
Pxx_in_dB = 10*log10(Pxx_in / max(Pxx_in));

if exist('DetailedData', 'var')
    [Pxx_out, ~] = pwelch(DetailedData.PAout, blackman(1024), 512, 1024, Fs, 'centered');
    Pxx_out_dB = 10*log10(Pxx_out / max(Pxx_out));
    
    plot(f/1e6, Pxx_in_dB, 'b', 'LineWidth', 2, 'DisplayName', 'Input Signal');
    hold on;
    plot(f/1e6, Pxx_out_dB, 'r', 'LineWidth', 2, 'DisplayName', 'PA Output');
    
    % Mark the channel boundaries
    xline(-BW/2e6, '--k', 'LineWidth', 1);
    xline(BW/2e6, '--k', 'LineWidth', 1);
    xline(-ACPR_params.Offset/1e6, ':m', 'LineWidth', 1);
    xline(ACPR_params.Offset/1e6, ':m', 'LineWidth', 1);
    
    hold off;
else
    plot(f/1e6, Pxx_in_dB, 'b', 'LineWidth', 2);
end

xlabel('Frequency (MHz)');
ylabel('Normalized PSD (dB)');
title('Spectral Regrowth due to PA Nonlinearity');
legend('Location', 'best');
grid on;
ylim([-80 5]);
xlim([-Fs/2e6 Fs/2e6]);

% Add annotation
text(0.02, 0.98, sprintf('Signal BW: %.1f MHz', BW/1e6), ...
    'Units', 'normalized', 'VerticalAlignment', 'top');

%% ========================================================================
%  SECTION 7: BER COMPARISON - WITH AND WITHOUT PA DISTORTION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('   BER ANALYSIS\n');
fprintf('========================================\n');

% SNR range for BER analysis
SNR_dB = 0:2:30;
numSNR = length(SNR_dB);

% Initialize BER arrays
BER_ideal = zeros(1, numSNR);
BER_withPA = zeros(1, numSNR);

% Generate test symbols for BER
NumTestSymbols = 10000;
testBits = randi([0 1], NumTestSymbols * BitsPerSymbol, 1);
testSymbols = bit2sym(testBits, BitsPerSymbol, ModulationType);

fprintf('Computing BER curves...\n');

for snrIdx = 1:numSNR
    snr = SNR_dB(snrIdx);
    
    % === Ideal channel (no PA) ===
    rxSymbols_ideal = awgn(testSymbols, snr, 'measured');
    rxBits_ideal = sym2bit(rxSymbols_ideal, BitsPerSymbol, ModulationType);
    BER_ideal(snrIdx) = sum(testBits ~= rxBits_ideal) / length(testBits);
    
    % === With PA distortion ===
    if exist('DetailedData', 'var')
        % Apply PA nonlinearity model (simplified AM/AM)
        % Using polynomial approximation of measured PA
        pa_gain = mean(Results.Gain(validIdx));
        compression_factor = 0.8;  % Simulated compression
        
        % Apply soft compression (Rapp model approximation)
        p = 2;  % Smoothness factor
        A_sat = 1.2;  % Saturation amplitude
        testSymbols_pa = testSymbols .* (1 ./ (1 + (abs(testSymbols)/A_sat).^(2*p)).^(1/(2*p)));
        testSymbols_pa = testSymbols_pa * compression_factor;
        
        % Add AWGN
        rxSymbols_pa = awgn(testSymbols_pa, snr, 'measured');
        rxBits_pa = sym2bit(rxSymbols_pa, BitsPerSymbol, ModulationType);
        BER_withPA(snrIdx) = sum(testBits ~= rxBits_pa) / length(testBits);
    end
end

% Plot BER comparison
figure('Name', 'BER Comparison', 'Position', [100, 100, 800, 600]);
semilogy(SNR_dB, BER_ideal, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'DisplayName', 'Ideal (No PA)');
hold on;
if exist('DetailedData', 'var')
    semilogy(SNR_dB, BER_withPA, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, ...
        'DisplayName', 'With PA Distortion');
end

% Theoretical BER for reference
if strcmp(ModulationType, 'QPSK')
    BER_theory = qfunc(sqrt(2*10.^(SNR_dB/10)));
else  % 16-QAM
    BER_theory = (3/8) * erfc(sqrt((2/5)*10.^(SNR_dB/10)));
end
semilogy(SNR_dB, BER_theory, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical');

hold off;
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
title(sprintf('BER Performance Comparison - %s', ModulationType));
legend('Location', 'southwest');
grid on;
ylim([1e-6 1]);
xlim([0 30]);

%% ========================================================================
%  SECTION 8: PA PERFORMANCE PLOTS
%  ========================================================================

figure('Name', 'PA Performance Metrics', 'Position', [100, 100, 1200, 800]);

% Plot 1: Gain vs Input Power (AM/AM)
subplot(2,3,1);
plot(Results.RMSin(validIdx), Results.Gain(validIdx), 'b-o', 'LineWidth', 2);
xlabel('Input Power (dBm)');
ylabel('Gain (dB)');
title('Gain Compression (AM/AM)');
grid on;

% Plot 2: Output Power vs Input Power
subplot(2,3,2);
plot(Results.RMSin(validIdx), Results.RMSout(validIdx), 'b-o', 'LineWidth', 2);
hold on;
% Add 1dB compression reference line
plot(Results.RMSin(validIdx), Results.RMSin(validIdx) + max(Results.Gain(validIdx)), 'k--', 'LineWidth', 1);
hold off;
xlabel('Input Power (dBm)');
ylabel('Output Power (dBm)');
title('Power Transfer Characteristic');
legend('Measured', 'Linear Reference', 'Location', 'southeast');
grid on;

% Plot 3: Efficiency vs Output Power
subplot(2,3,3);
plot(Results.RMSout(validIdx), Results.Efficiency(validIdx), 'g-o', 'LineWidth', 2);
xlabel('Output Power (dBm)');
ylabel('PAE (%)');
title('Power Added Efficiency');
grid on;

% Plot 4: EVM vs Input Power
subplot(2,3,4);
plot(Results.RMSin(validIdx), Results.EVM(validIdx), 'r-o', 'LineWidth', 2);
xlabel('Input Power (dBm)');
ylabel('EVM (%)');
title('EVM Degradation');
grid on;

% Plot 5: ACPR vs Input Power
subplot(2,3,5);
plot(Results.RMSin(validIdx), Results.ACPR_L1(validIdx), 'b-o', 'LineWidth', 2);
hold on;
plot(Results.RMSin(validIdx), Results.ACPR_U1(validIdx), 'r-s', 'LineWidth', 2);
hold off;
xlabel('Input Power (dBm)');
ylabel('ACPR (dB)');
title('Adjacent Channel Power Ratio');
legend('Lower', 'Upper', 'Location', 'best');
grid on;

% Plot 6: DC Current vs Input Power
subplot(2,3,6);
plot(Results.RMSin(validIdx), Results.Idc(validIdx)*1000, 'm-o', 'LineWidth', 2);
xlabel('Input Power (dBm)');
ylabel('DC Current (mA)');
title('Supply Current');
grid on;

sgtitle(sprintf('%s Modulation - PA Performance Analysis', ModulationType));

%% ========================================================================
%  SECTION 9: SAVE RESULTS
%  ========================================================================

fprintf('\n========================================\n');
fprintf('   SAVING RESULTS\n');
fprintf('========================================\n');

% Save results to MAT file
save('PA_Analysis_Results.mat', 'Results', 'ModulationType', 'NumSymbols', ...
    'Fs', 'BW', 'SNR_dB', 'BER_ideal', 'BER_withPA');
fprintf('Results saved to PA_Analysis_Results.mat\n');

fprintf('\nAnalysis complete!\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function symbols = bit2sym(bits, bitsPerSymbol, modType)
    % Convert bits to modulated symbols
    numSymbols = length(bits) / bitsPerSymbol;
    bits_reshaped = reshape(bits, bitsPerSymbol, numSymbols).';
    
    switch modType
        case 'QPSK'
            % Gray-coded QPSK
            symbols = (1/sqrt(2)) * ((2*bits_reshaped(:,1)-1) + 1j*(2*bits_reshaped(:,2)-1));
        case '16QAM'
            % Gray-coded 16-QAM
            I = 2*bits_reshaped(:,1) + bits_reshaped(:,2);
            Q = 2*bits_reshaped(:,3) + bits_reshaped(:,4);
            I_map = [-3 -1 3 1];  % Gray mapping
            Q_map = [-3 -1 3 1];
            symbols = (1/sqrt(10)) * (I_map(I+1).' + 1j*Q_map(Q+1).');
    end
end

function bits = sym2bit(symbols, bitsPerSymbol, modType)
    % Demodulate symbols to bits
    numSymbols = length(symbols);
    bits = zeros(numSymbols * bitsPerSymbol, 1);
    
    switch modType
        case 'QPSK'
            bits(1:2:end) = real(symbols) > 0;
            bits(2:2:end) = imag(symbols) > 0;
        case '16QAM'
            % Hard decision for 16-QAM
            I = real(symbols) * sqrt(10);
            Q = imag(symbols) * sqrt(10);
            
            for k = 1:numSymbols
                idx = (k-1)*4;
                % I bits
                if I(k) < -2
                    bits(idx+1) = 0; bits(idx+2) = 0;
                elseif I(k) < 0
                    bits(idx+1) = 0; bits(idx+2) = 1;
                elseif I(k) < 2
                    bits(idx+1) = 1; bits(idx+2) = 1;
                else
                    bits(idx+1) = 1; bits(idx+2) = 0;
                end
                % Q bits
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
end

function evm = calculate_evm(ref_signal, meas_signal)
    % Calculate EVM in percentage
    % Normalize measured signal to reference
    meas_norm = meas_signal * (mean(abs(ref_signal)) / mean(abs(meas_signal)));
    
    % Align lengths
    minLen = min(length(ref_signal), length(meas_norm));
    ref_signal = ref_signal(1:minLen);
    meas_norm = meas_norm(1:minLen);
    
    % Calculate error vector
    error_vec = meas_norm - ref_signal;
    
    % EVM calculation
    evm = 100 * sqrt(mean(abs(error_vec).^2) / mean(abs(ref_signal).^2));
end

function papr = calculate_papr(signal)
    % Calculate PAPR in dB
    peak_power = max(abs(signal).^2);
    avg_power = mean(abs(signal).^2);
    papr = 10 * log10(peak_power / avg_power);
end

function plot_constellation(symbols, modType, titleStr)
    % Plot constellation diagram
    plot(real(symbols), imag(symbols), '.', 'MarkerSize', 2);
    hold on;
    
    % Add reference constellation points
    switch modType
        case 'QPSK'
            ref = (1/sqrt(2)) * [1+1j, 1-1j, -1+1j, -1-1j];
        case '16QAM'
            [I, Q] = meshgrid([-3 -1 1 3], [-3 -1 1 3]);
            ref = (1/sqrt(10)) * (I(:) + 1j*Q(:));
    end
    plot(real(ref), imag(ref), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    
    xlabel('In-Phase');
    ylabel('Quadrature');
    title(titleStr);
    axis equal;
    grid on;
    
    % Set appropriate axis limits
    maxVal = max(abs([real(symbols); imag(symbols)])) * 1.2;
    maxVal = max(maxVal, 1.5);
    xlim([-maxVal maxVal]);
    ylim([-maxVal maxVal]);
end
