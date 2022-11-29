function [lats_gsInSeq_dist, lats_gsInSeq_med, no_dp] = infSampling_getLatFirstGs2(latencis, gsNos, gsOnSOI, trialNos, noOI, setSizes)

    % Get latency distributions and median latency of specific gaze shifts
    % in a sequence of gaze shifts that landed or did not land on a
    % provided stimulus category of interest
    % CALCULATE LATENCIES FIRST FOR SET SIZES THEN AVERAGE
    % Input
    % latencis:          column vector with latencies of all detected gaze
    %                    shifts
    % gsNos:             column vector with position of gaze shifts in a
    %                    sequence of gaze shifts (i.e., first, second,
    %                    third, ... gaze shift in trial)
    % gsOnSOI:           for which gaze shift in a sequence do we want to
    %                    get latencies
    % trialNos:          trialnumber belonging to a latency
    % noOI:              # of gaze shift in trial we want to analyse
    % setSizes:          # easy distractors
    % Output
    % lats_gsInSeq_dist: latency distributions of gaze-shifts-of-interest
    %                    that landed and did not land on a stimulu category
    %                    of interest
    %                    (:, 1): gaze shift landed ("1") or not ("0") on
    %                            stimulus category of interest
    %                    (:, 2): gaze shift latency
    %                    (:, 3): trialnumber of gaze shift
    % lats_gsInSeq_med:  median latencies
    %                    (1): gaze shifts to stimulus category of interest
    %                    (2): gaze shifts not to stimulus category of
    %                         interest
    % no_dp:             number of datapoints we used to calculate the
    %                    output latency values

    %% Get size size conditions
    ssCat = unique(setSizes(~isnan(setSizes)));
    NOSS  = size(ssCat, 1);


    %% Latencies for set size conditions
    temp    = NaN(NOSS, 2);
    latDist = cell(1, NOSS);
    for ss = 1:NOSS % Set sizes

        % Get latencies and # datapoints in each category
        % We extract latencies of gaze shifts with a specific position "noOI" in
        % a sequence of gaze shifts. Latencies are extracted seperately for gaze
        % shifts that landed or did not land on the stimulus category of
        % interested (as indicated by the vector "gsOnSOI")
        li_lat_gsOnSOI    = gsNos == noOI & any(setSizes == ssCat(ss, :), 2) & gsOnSOI == 1;
        li_lat_gsNotOnSOI = gsNos == noOI & any(setSizes == ssCat(ss, :), 2) & gsOnSOI == 0;

        % Get latency distributions
        % Latencies of gaze-shifts-of-interest that landed on the stimulus
        % category of interest are labeled with "1", gaze-shifts-of-interest
        % that did not land on the stimulus category of interest are labeled
        % with "0"
        no_gsOnSOI    = sum(li_lat_gsOnSOI);
        no_gsNotOnSOI = sum(li_lat_gsNotOnSOI);
        no_dp         = [no_gsOnSOI no_gsNotOnSOI];

        lat_gsOnSOI    = [ones(no_gsOnSOI, 1)     latencis(li_lat_gsOnSOI)    trialNos(li_lat_gsOnSOI)];
        lat_gsNotOnSOI = [zeros(no_gsNotOnSOI, 1) latencis(li_lat_gsNotOnSOI) trialNos(li_lat_gsNotOnSOI)];

        latDist{ss} = sortrows([lat_gsOnSOI; lat_gsNotOnSOI], 3);

        % Get MEDIAN latency
        temp(ss, :) = [median(latencis(li_lat_gsOnSOI), 'omitnan') ...
                       median(latencis(li_lat_gsNotOnSOI), 'omitnan')];

    end


    %% Output
    lats_gsInSeq_med  = mean(temp, 1, 'omitnan');         % MEAN latency
    lats_gsInSeq_dist = sortrows(vertcat(latDist{:}), 3); % Latency distributions

end