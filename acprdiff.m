function ACPRdiff=acprdiff(ACPRref,ACPR)
% ACPRdiff=acprdiff(ACPRref,ACPR)
% This function takes two ACPR structures, and computes the differences in
% each band. The first argument is considered as a reference.
%
% (c) Mazen Abi Hussein, 2012

ACPRdiff.L1=ACPRref.L1-ACPR.L1;
ACPRdiff.U1=ACPRref.U1-ACPR.U1;
ACPRdiff.L2=ACPRref.L2-ACPR.L2;
ACPRdiff.U2=ACPRref.U2-ACPR.U2;
