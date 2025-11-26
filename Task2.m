%% ======================================================
%  TELECOM PROJECT - TASK II
%  Noise Robustness & Modulation Comparison
%  Author: Nom Prénom
%  ======================================================

clc; clear; close all;

%% 1. Paramètres généraux
N = 1e5;                         % Nombre de bits
Rs = 1e5;                        % Débit symbole
Fs = Rs * 8;                     % Fréquence d'échantillonnage
sps = Fs / Rs;                   % Échantillons par symbole
rolloff = 0.25; span = 10;
rrcFilter = rcosdesign(rolloff, span, sps, 'sqrt');
EbN0_dB_vec = -2:2:20;           % Plage de SNR à tester

modSchemes = {'BPSK','QPSK','16QAM'};
colors = {'b','r','g'};

%% ======================================================
%  2. FONCTION MODULATION / DEMODULATION
% ======================================================
getModulator = @(scheme) switchModulator(scheme);

%% ======================================================
%  3. CAS 1 : Canal AWGN
% ======================================================
figure('Name','BER vs SNR - Canal AWGN','Color','w');
for m = 1:length(modSchemes)
    scheme = modSchemes{m};
    [modFun, demodFun, M] = getModulator(scheme);
    BER = testChannel(N, M, klog(M), rrcFilter, sps, EbN0_dB_vec, ...
                      modFun, demodFun, 'awgn');
    semilogy(EbN0_dB_vec, BER, 'Color', colors{m}, 'LineWidth', 1.5);
    hold on;
end
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('Comparaison BPSK / QPSK / 16QAM - Canal AWGN');
legend(modSchemes, 'Location','southwest');

%% ======================================================
%  4. CAS 2 : Canal AWGN + Bruit impulsif
% ======================================================
figure('Name','BER vs SNR - AWGN + Impulsif','Color','w');
for m = 1:length(modSchemes)
    scheme = modSchemes{m};
    [modFun, demodFun, M] = getModulator(scheme);
    BER = testChannel(N, M, klog(M), rrcFilter, sps, EbN0_dB_vec, ...
                      modFun, demodFun, 'impulsive');
    semilogy(EbN0_dB_vec, BER, 'Color', colors{m}, 'LineWidth', 1.5);
    hold on;
end
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('Comparaison BPSK / QPSK / 16QAM - AWGN + Bruit impulsif');
legend(modSchemes, 'Location','southwest');

%% ======================================================
%  5. CAS 3 : Canal de Rayleigh
% ======================================================
figure('Name','BER vs SNR - Canal de Rayleigh','Color','w');
for m = 1:length(modSchemes)
    scheme = modSchemes{m};
    [modFun, demodFun, M] = getModulator(scheme);
    BER = testChannel(N, M, klog(M), rrcFilter, sps, EbN0_dB_vec, ...
                      modFun, demodFun, 'rayleigh');
    semilogy(EbN0_dB_vec, BER, 'Color', colors{m}, 'LineWidth', 1.5);
    hold on;
end
grid on;
xlabel('E_b/N_0 (dB)'); ylabel('BER');
title('Comparaison BPSK / QPSK / 16QAM - Canal Rayleigh');
legend(modSchemes, 'Location','southwest');

disp('Simulation complète terminée pour les trois canaux.');

%% ======================================================
%  FONCTIONS LOCALES
% ======================================================

function [modFun, demodFun, M] = switchModulator(scheme)
    switch scheme
        case 'BPSK'
            M = 2;
            modFun = @(x) pskmod(x, M, 0, 'InputType','bit');
            demodFun = @(x) pskdemod(x, M, 0, 'OutputType','bit');
        case 'QPSK'
            M = 4;
            modFun = @(x) pskmod(x, M, pi/M, 'InputType','bit');
            demodFun = @(x) pskdemod(x, M, pi/M, 'OutputType','bit');
        case '16QAM'
            M = 16;
            modFun = @(x) qammod(x, M, 'InputType','bit', 'UnitAveragePower', true);
            demodFun = @(x) qamdemod(x, M, 'OutputType','bit', 'UnitAveragePower', true);
        otherwise
            error('Schéma de modulation inconnu.');
    end
end

function k = klog(M)
    k = log2(M);
end

function BER = testChannel(N, M, k, rrcFilter, sps, EbN0_dB_vec, modFun, demodFun, mode)
    symbolsPerTest = N/k;
    bits = randi([0 1], N, 1);
    symbols = modFun(bits);
    tx_signal = upfirdn(symbols, rrcFilter, sps, 1);
    delay = 10;
    BER = zeros(size(EbN0_dB_vec));

    for i = 1:length(EbN0_dB_vec)
        EbN0_dB = EbN0_dB_vec(i);
        snr = EbN0_dB + 10*log10(k);
        rx_signal = awgn(tx_signal, snr, 'measured');

        switch mode
            case 'impulsive'
                num_impulses = 100;
                idx = randperm(length(rx_signal), num_impulses);
                noiseAmp = 1 / sqrt(10^(EbN0_dB/10)); % amplitude adaptative
                rx_signal(idx) = rx_signal(idx) + noiseAmp * randn(num_impulses,1);
            case 'rayleigh'
                h = (randn(size(rx_signal)) + 1j*randn(size(rx_signal)))/sqrt(2);
                rx_signal = rx_signal .* h;
                rx_signal = rx_signal ./ h; % égalisation parfaite
        end

        % Réception
        rx_filtered = upfirdn(rx_signal, rrcFilter, 1, sps);
        rx_sync = rx_filtered(delay+1 : delay+length(symbols));
        bits_rx = demodFun(rx_sync);

        L = min(length(bits), length(bits_rx));
        BER(i) = mean(bits(1:L) ~= bits_rx(1:L));
    end
end
