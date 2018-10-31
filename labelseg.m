%Segregates data so that a given ratio of labels results, here it's
%hardcoded to have a 7:1 ratio of label 1:label 0 (0 or specified number).
%This function reduces the data and label matrix size so that the specified
%ratio results in the shortened data and label matrix outputs, and keeps data
%associated with each label by row.
%
%Output1 is label output, output2 is data output
%Made by Kevin Walsh, MOCORE Lab, 10/30/2018

function [output1, output2] = labelseg(labels, data, optionalVarZeroLabelValue);
if nargin > 2;
  defaultParam = optionalVarZeroLabelValue;
else
  defaultParam = 0;
end
lcz = 0;
lci = 0;
    c = 1;
    d = 1;
for i = 1:length(labels);
    if labels(i) == defaultParam;
        lcz = lcz + 1;
    end
    if labels(i) ~= defaultParam;
            lci = lci + 1;
    end
end 
newzero = round(lci/7);
    newsize = (lci + newzero);
    output1 = zeros(newsize,1);
    output2 = zeros(newsize, size(data, 2));  
if (lci/7) < lcz;
    for i = 1:length(labels);
        if c < newsize;
            if labels(i) ~= defaultParam;
                output1(c) = labels(i);
                output2(c,:) = data(i,:);
                c = c + 1;
            if labels(i) == defaultParam;
                    if d < newzero;
                        output1(c) = labels(i);
                        output2(c,:) = data(i,:);
                        c = c + 1;
                        d = d + 1;
                    end
            end      
            end
        end       
    end
elseif (lci/7) >= lcz;
    output1 = labels;
    output2 = data;
end
end
% 
% rowToInsert = 3 % Whatever you want.
% C = [A(1:rowToInsert-1, :); B; A(rowToInsert:end, :)]
