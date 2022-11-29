function [out_propChoices, out_fixNum] = test_unsystematic_fixations_2(sze_all, gain, freeParameter)

    % 1: number of targets in set
    % 2: probability to choose set
    % 3: absolute probability to find target in set
    % 4: absolute probability to find target in this fixation (2*3)
    % 5: cumulative probability that target has not been found yet
    % 6: cumulative probability that target has been found
    % 7: conditional probability to find target in this fixation
    % 8-9: doesn't work
    % 10: cumulative number of fixations in set


    %% Init
    % Total number of elements
    n = unique(sum(sze_all, 2));
    % Preference for one set
%     pref = 1.9; % 0-1: pref 1; 1-2: pref 2;
    % acc = [0.9 0.7];
    % Size of sets
%     sze = [2 8];%[2 8];
    noSs = size(sze_all, 1);


    %% Translate relative gain into choice bias
    gain_relative = gain(:, 2) - gain(:, 1);
%         gain_relative = gain(:, 1) - gain(:, 2);
    pref_all      = cdf('Normal', gain_relative, 0, freeParameter) .* 2;
%     pref_all      = cdf('Normal', gain_relative, 0, freeParameter);


    %%
    out_fixNum      = NaN(noSs, 3, 2);
    out_propChoices = NaN(noSs, 2);
    for ss = 1:noSs % Set size

        % 
        sze  = sze_all(ss, :);
        pref = pref_all(ss);

        % initialize sets
        data1 = [];
        data2 = [];
        data3 = [];
        % number of remaining elements in sets
        data1(1,1) = sze(1);
        data2(1,1) = sze(2);
%         data1(1,1) = sze(1) * pref;
%         data2(1,1) = sze(2) * (1 - pref);
        data3(1,1) = data1(1,1)+data2(1,1);
        % probability to choose set
%         if pref<1                                                                     % Adjust all fixations
%             data1(1,2) = data1(1,1)./(data1(1,1)+pref*data2(1,1));
%             data2(1,2) = (pref*data2(1,1))./(data1(1,1)+pref*data2(1,1));
%         else    
%             data1(1,2) = ((2-pref)*data1(1,1))./((2-pref)*data1(1,1)+data2(1,1));
%             data2(1,2) = data2(1,1)./((2-pref)*data1(1,1)+data2(1,1));    
%         end
        data1(1,2) = data1(1,1)./(data1(1,1)+1*data2(1,1));                           % Do not adjust first fixation
        data2(1,2) = (1*data2(1,1))./(data1(1,1)+1*data2(1,1));
%         data1(1,2) = data1(1,1) ./ (data1(1,1) + data2(1,1));                       % Ignore set size
%         data2(1,2) = data2(1,1) ./ (data1(1,1) + data2(1,1));
%         data1(1,2) = data1(1,1) ./ ((pref * data1(1,1)) + ((1 - pref) * data2(1,1)));                       % Ignore set size
%         data2(1,2) = data2(1,1) ./ ((pref * data1(1,1)) + ((1 - pref) * data2(1,1)));
        data3(1,2) = NaN;%1;
        % absolute probability to find target when set is fixated
        data1(1,3) = 1./data1(1,1);
        data2(1,3) = 1./data2(1,1);
        data3(1,3) = NaN;%2./data3(1,1);
        % absolute probability to find target in this fixation
        data1(1,4) = data1(1,2)*data1(1,3);
        data2(1,4) = data2(1,2)*data2(1,3);
        data3(1,4) = data1(1,4)+data2(1,4);%data3(1,2)*data3(1,3);
        % wrong addition of probabilities
        data3(1,8) = data1(1,4)+data2(1,4);
        data3(1,9) = 1-data3(1,8);
        % data3(1,4) = 1-(data1(1,4)*data2(1,4));
        % cumulative probability that target has not been found yet
        data1(1,5) = 1-data1(1,4);
        data2(1,5) = 1-data2(1,4);
        data3(1,5) = 1-data3(1,4);
        % cumulative probability that target has been found
        data1(1,6) = 1-data1(1,5);
        data2(1,6) = 1-data2(1,5);
        data3(1,6) = 1-data3(1,5);
        % conditional probability to find target in this fixation (given it has not yet
        % been found)
        data1(1,7) = data1(1,4);
        data2(1,7) = data2(1,4);
        data3(1,7) = data3(1,4);
        % cumulative number of fixations in set
        data1(1,10) = data1(1,2);
        data2(1,10) = data2(1,2);

        % 
        % data3(1,5) = (1-data2(1,4)).*(1-data1(1,4));
        for f=2:n
            % update remaining number of elements
            data1(f,1) = max([0 data1(f-1,1)-data1(f-1,2)]);
            data2(f,1) = max([0 data2(f-1,1)-data2(f-1,2)]);
%             data1(f,1) = max([0 data1(f-1,1)-data1(f-1,2)]) * pref;
%             data2(f,1) = max([0 data2(f-1,1)-data2(f-1,2)]) * (1 - pref);
            data3(f,1) = data1(f,1)+data2(f,1);%max([0 data3(f-1,1)-data3(f-1,2)]);
            if (data1(f,1)==0)||(data2(f,1)==0) % Break if no elements left to fixate in any set
                data1 = data1(1:end-1,:);
                data2 = data2(1:end-1,:);
                data3 = data3(1:end-1,:);
                break;
            end

            % probability to choose set
            if pref<1
                data1(f,2) = data1(f,1)./(data1(f,1)+pref*data2(f,1));
                data2(f,2) = (pref*data2(f,1))./(data1(f,1)+pref*data2(f,1)); 
            else
                data1(f,2) = ((2-pref)*data1(f,1))./((2-pref)*data1(f,1)+data2(f,1));
                data2(f,2) = data2(f,1)./((2-pref)*data1(f,1)+data2(f,1));        
            end
%             data1(f,2) = data1(f,1) ./ (data1(f,1) + data2(f,1));             % Ignore set size
%             data2(f,2) = data2(f,1) ./ (data1(f,1) + data2(f,1));
%             data1(f,2) = pref;% data1(f,1) ./ ((pref * data1(f,1)) + ((1 - pref) * data2(f,1)));                       % Ignore set size
%             data2(f,2) = 1-pref;% data2(f,1) ./ ((pref * data1(f,1)) + ((1 - pref) * data2(f,1)));
            data3(f,2) = NaN;%1;
            % absolute probability to find target when set is fixated
            data1(f,3) = 1./data1(f,1);%min([1 1./data1(f,1)]);
            data2(f,3) = 1./data2(f,1);%min([1 1./data2(f,1)]); 
            data3(f,3) = NaN;%2./data3(f,1); 
            % absolute probability to find target in this fixation
            data1(f,4) = data1(f,2)*data1(f,3);
            data2(f,4) = data2(f,2)*data2(f,3); 
            %data3(f,4) = data3(f,2)*data3(f,3);    
            data3(f,4) = data1(f,4)+data2(f,4);
        %     data3(f,4) = 1-(data1(f,4)*data2(f,4));
            % wrong
            data3(f,8) = min([1 data1(f,4)+data2(f,4)]);
            data3(f,9) = 1-data3(f,8);

            tmp1 = 1;
            tmp2 = 1;
            tmp3 = 1;
            for f2=1:f
                tmp1 = tmp1 .* (1-data1(f2,4));
                tmp2 = tmp2 .* (1-data2(f2,4));
                tmp3 = tmp3 .* (1-data3(f2,4));
            end
            % cumulative probability that target has not been found yet
            data1(f,5) = tmp1;
            data2(f,5) = tmp2;
            data3(f,5) = tmp3;
            % cumulative probability that target has been found 
            data1(f,6) = 1-data1(f,5);
            data2(f,6) = 1-data2(f,5);
            data3(f,6) = 1-data3(f,5);
            % conditional probability to find target in this fixation (given it has not yet
            % been found)
            data1(f,7) = data1(f,4).*data3(f-1,5);
            data2(f,7) = data2(f,4).*data3(f-1,5);
            data3(f,7) = data3(f,4).*data3(f-1,5);
        %     tmp = 1;
        %     for f2=1:f
        %         tmp = tmp .* (1-data2(f2,4));
        %     end
        %     data2(f,5) = tmp;   
        %     tmp = 1;
        %     for f2=1:f
        %         tmp = tmp .* (1-data2(f2,4)) .* (1-data1(f2,4));
        %     end
        %     data3(f,5) = tmp;  

           % cumulative number of fixations in set
           data1(f,10) = data1(f,2)+data1(f-1,10);
           data2(f,10) = data2(f,2)+data2(f-1,10);
        end
        % number of fixation until target is found
        fixNum3 = sum(data3(:,7).*(1:size(data3,1))');
        % fixations in sets
        fixNum1 = sum(data1(:,10).*data3(:,7));
        fixNum2 = sum(data2(:,10).*data3(:,7));
%         fprintf('Fixations (Set1, Set2, both, check):\t%.2f,\t%.2f,\t%.2f,\t%.2f\n',fixNum1,fixNum2,fixNum3,fixNum1+fixNum2);

        tg3 = sum(data3(:,7));
        tg1 = sum(data1(:,7));
        tg2 = sum(data2(:,7));
%         fprintf('Targets (Set1, Set2, both, check):\t\t%.2f,\t%.2f,\t%.2f,\t%.2f\n',tg1,tg2,tg3,tg1+tg2);
        % sum((data1(:,10)+data2(:,10)).*data3(:,7))

        % Number fixations on chosen/not-chosen set
        nonChosen = sum(data1(:, 7) .* data2(:, 10)) + sum(data2(:, 7) .* data1(:, 10));
        chosen    = sum(data1(:, 7) .* data1(:, 10)) + sum(data2(:, 7) .* data2(:, 10));
%         fprintf('Fixations (Chosen, Nonchosen, both, check):\t%.2f,\t%.2f,\t%.2f,\t%.2f\n',chosen,nonChosen,fixNum3,chosen+nonChosen);
        % sum((data1(:,10)+data2(:,10)).*data3(:,7))

        % Output
        out_fixNum(ss, :, 1)   = [fixNum1 fixNum2 sum([fixNum1 fixNum2])];
        out_fixNum(ss, :, 2)   = [chosen nonChosen sum([chosen nonChosen])];
        out_propChoices(ss, :) = [tg1 tg2];

    end

end