%% Get subject number and experiment number
subject = input('Subject: ');
expNo   = input('Experiment: ');


%% Read .log file
pathToDataFolder = 'C:/src/ilja/1_informationSampling/2_code_limitedViewingTime/data/'; % Fill in path to experiment on lab computer
pathToData       = strcat(pathToDataFolder, 'e', num2str(expNo), 'v', num2str(subject), 'b1');
cd(pathToData);

logFileName = strcat('e', num2str(expNo), 'v', num2str(subject), 'b1', '.log');
logFile     = dlmread(logFileName);


%% Get and output final score
finalScore = logFile(end, 37);
strcat('This participants score is:', num2str(finalScore))


%% Return to root
cd ../..