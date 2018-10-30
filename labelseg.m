function [output1, output2] = labelseg(labels, data, optionalVarZeroLabelValue)
if nargin > 1
  defaultParam = optionalVarZeroLabelValue;
else
  defaultParam = 0;
end
dc = 0;
lcz = 0;
lci = 0;
for i = 1:length(labels)
    if labels(i) == defaultParam;
        lcz = lcz + 1;
    end
    if labels(i) ~= defaultParam;
            lci = lci + 1;
    end
end 
if (lci/7) > lci
    newzero = round(lci/7)
    newsize = (lci + newzero)
    output1 = zeros(1, newsize)
    output2 = zeros(newsize, size(data, 2))   
    c = 0
    for i = 1:length(labels)
        if i ~= defaultParam
            output1(i) == labels(i)
            output2(i,:) == data(i,:)
            c = c + 1;
        if i == defaultParam
            output1(i) == labels(i)
            output2(i,:) == data(i,:)
            c = c + 1;
    end
else
    output1 = labels
    output2 = data
end
end

rowToInsert = 3 % Whatever you want.
C = [A(1:rowToInsert-1, :); B; A(rowToInsert:end, :)]
