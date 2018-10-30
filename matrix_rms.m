%applies Signal_RMS to columns of a matrix
function y = matrix_rms(input, windowlength, overlap, zeropad)
for i = 1:size(input,2);
    y(:,i) = signal_rms(input(:,i),windowlength,0,0);
end
end