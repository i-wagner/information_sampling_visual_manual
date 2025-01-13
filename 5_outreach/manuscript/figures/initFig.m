folder.root = '/Users/ilja/Library/CloudStorage/GoogleDrive-ilja.wagner1307@gmail.com/My Drive/mr_informationSamplingVisualManual/';
folder.data = strcat(folder.root, '2_data/');

load(strcat(folder.data, 'data_newPipeline.mat')); % Data from experiments
opt_visuals; % Figure visuals