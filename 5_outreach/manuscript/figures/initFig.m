folder.root = '/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/';
folder.data = strcat(folder.root, '2_data/');

load(strcat(folder.data, 'data_newPipeline.mat')); % Data from experiments
opt_visuals; % Figure visuals