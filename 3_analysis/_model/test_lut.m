clear all; close all; clc
profile off


%% Init
setSizes = [(1:9)' (9:-1:1)'];
p_lvl    = (1:4);

NOISESAMPLES = 10000;
SETSIZES     = size(setSizes, 1);
PLVLNO       = numel(p_lvl);
SUBJECTS     = 19;


%% Simulate
sumChoice_avg    = NaN(SUBJECTS, 2, SETSIZES, PLVLNO);
sumFixSet_avg    = NaN(SUBJECTS, 2, SETSIZES, PLVLNO);
sumFixChoice_avg = NaN(SUBJECTS, 2, SETSIZES, PLVLNO);
time_it          = NaN(SUBJECTS, PLVLNO);
% biasSetSize      = NaN(1, SETSIZES);
% idx              = 1;
for sub = 1:SUBJECTS % Subject

%     rng(30);
    noise         = randn(1, NOISESAMPLES) .* 0.20;
    gain          = rand(SETSIZES, 2);
    gain_relative = repmat(gain(:, 2) - gain(:, 1), 1, NOISESAMPLES);
    gain_relative = gain_relative + noise;
    for p = 1:PLVLNO % Precision

        % Load lookup-table
        precisions   = p_lvl(p);                                                       
        biasSetSize  = load(['infSampling_lut_', num2str(precisions) ,'.mat']);
        biasSetSize  = biasSetSize.lut;
        UNIQUEBIASES = numel(0:10^-precisions:2);
        BIASSTARTROW = 0:UNIQUEBIASES:(size(biasSetSize, 1)-UNIQUEBIASES);

        % Calculate empirical bias
        bias_all = round(cdf('Normal', gain_relative, 0, 0.30) .* 2, precisions);

        % Predict
        clc; tic
%         profile on
        sumChoice    = NaN(NOISESAMPLES, 2, SETSIZES);
        sumFixSet    = NaN(NOISESAMPLES, 2, SETSIZES);
        sumFixChoice = NaN(NOISESAMPLES, 2, SETSIZES);
        for ns = 1:NOISESAMPLES

%             disp(['Now calculating noise sample # ', num2str(ns)]);
            for ss = 1:size(setSizes, 1) % Set size

                % Get bias parameter for current set size "ss"
                setSize_single = setSizes(ss, :);
                bias_single    = bias_all(ss, ns);

                % Use recursive algorithm to predict
                entry = numel(0:10^-precisions:bias_single)+BIASSTARTROW(setSize_single(1));                  % Fastest
%                 entry2 = findRowFast(biasSetSize(:, 1:3), [bias_single setSize_single]);         % Intermediate (.mex)
%                 if entry ~= entry2; keyboard; end
    %             entry = all(bsxfun(@eq, [bias_single setSize_single], biasSetSize(:, 1:3)), 2); % Slower
    %             entry = all([biasSetSize(:, 1) == bias_single, ...                                Even slower
    %                          biasSetSize(:, 2) == setSize_single(1), ...
    %                          biasSetSize(:, 3) == setSize_single(2)], 2);
    %             entry = ismember(biasSetSize(:, 1:3), [bias_single setSize_single], 'rows');    % Slowest
                if ~isnan(entry)

                    sumChoice(ns, :, ss)    = biasSetSize(entry, 4:5);
                    sumFixSet(ns, :, ss)    = biasSetSize(entry, 6:7);
                    sumFixChoice(ns, :, ss) = biasSetSize(entry, 8:9);

                else

                    [allProb, allFix] = recursiveProb(bias_single, setSize_single);
                    for s = 1:2

                        sumChoice(ns, s, ss) = sum(allProb{s});
                        sumFixSet(ns, s, ss) = sum([allFix{1, s}.*allProb{1} allFix{2, s}.*allProb{2}]);

                    end
                    sumFixChoice(ns, 1, ss) = sum([allFix{1,1}.*allProb{1} allFix{2,2}.*allProb{2}]);
                    sumFixChoice(ns, 2, ss) = sum([allFix{1,2}.*allProb{1} allFix{2,1}.*allProb{2}]);

                    biasSetSize(idx, :) = [bias_single ...
                                           setSize_single ...
                                           sumChoice(ns, :, ss) ...
                                           sumFixSet(ns, :, ss) ...
                                           sumFixChoice(ns, :, ss)];
                    idx = idx + 1;

                end

            end

        end
        sumChoice_avg(sub, :, :, p)    = mean(sumChoice, 1);
        sumFixSet_avg(sub, :, :, p)    = mean(sumFixSet, 1);
        sumFixChoice_avg(sub, :, :, p) = mean(sumFixChoice, 1);

        time_it(sub, p) = toc;
%         profile viewer

    end

end


%% Plot
close all
subplot(1, 4, 1)
plot(1:PLVLNO, mean(time_it, 1), ...
     '-o')
axis([0 PLVLNO+1 0 ceil(max(time_it(:)))])
xticks(1:1:PLVLNO)
yticks(0:0.50:ceil(max(time_it(:))))
xlabel('Precision');
ylabel('Duration single subject [s]');
title('Duration single subject = 90,0000 loop iterations')
box off

dat    = {sumChoice_avg; sumFixSet_avg; sumFixChoice_avg};
labs_y = {'Choices easy [proportion]'; 'Fixations on easy set [#]'; 'Fixations on chosen set [#]'};
jitter = linspace(-0.75, 0.75, PLVLNO);
for sp = 2:4

    subplot(1, 4, sp)
    hold on
    for p = 1:PLVLNO

        plot((1:9)+jitter(p), squeeze(mean(dat{sp-1}(:, 1, :, p), 1)), ...
             'o')
        axis([0 10 0 1])
        box off

    end
    for ss = 1:9

        plot(ss+jitter(1:end), squeeze(mean(dat{sp-1}(:, 1, ss, 1:PLVLNO), 1)), ...
             '-', ...
             'Color', [0 0 0])

    end
    hold off
    axis([0 10 min(dat{sp-1}(:)) max(dat{sp-1}(:))])
    ylabel(labs_y{sp-1});
    h_leg = legend({'1' '2' '3' '4'});
    legend box off
    title(h_leg, 'Precision')

end