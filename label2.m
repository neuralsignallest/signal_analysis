% label data using threshold after calculating moving average
% return labeled data
% by Norika 

function label_data = label1(loaded_data,label_class,cutoff)
  
abs_data = abs(loaded_data);     % Rectified data

for mus = 1:7;
    MAV(:,mus) = movingmean(abs_data(:,mus), 80);
end

sum_muscle_data = sum(MAV,2);

label_data = [];

for i = 1:length(sum_muscle_data);
    if sum_muscle_data(i,1) > cutoff
        label_data = [label_data;label_class];
    else 
        label_data = [label_data;7];
    end
end




