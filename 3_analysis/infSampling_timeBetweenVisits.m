function aoi_enterLeave_out = infSampling_timeBetweenVisits(aoi_enterLeave, minDist_leaveReentry)

    % Check if reasonable amount of time passed between leaving on AOI and
    % entering the subsequent AOI; merge instances, in which the visit time
    % is beneath the criterion. ONLY FOR MULTIPLE VISITS OF THE SAME AOI
    % Input
    % aoi_enterLeave:        matrix, with number of datasample when AOI was
    %                        entered (:, 1) and when it was left (:, 2).
    %                        Rows are different AOI visits. AOI VISITS HAVE
    %                        TO BE SORTED IN ASCENDING ORDER
    % minDist_leaveReentry:  integer, representing the minimum time that has
    %                        to pass between leaving one AOI and entering
    %                        the  next AOI
    % Output
    % aoi_enterLeave_out:    same as input, but instances with inssuficient
    %                        time between subsequent visits are merged

    %% Sort AOI visits in ascending order
    aoi_enterLeave_out = aoi_enterLeave;
    aoi_enterLeave_out = sortrows(aoi_enterLeave_out, 1);


    %% Check if reasonable time passed between subsequent AOI visits
    it_wl = 1;
    while 1

        no_aoiEntries = size(aoi_enterLeave_out, 1);
        if no_aoiEntries == 1 || it_wl == no_aoiEntries

            clear no_aoiEntries it_wl
            break

        end

        offOn_consec = diff([aoi_enterLeave_out(it_wl, 2) aoi_enterLeave_out(it_wl+1, 1)]);
        if offOn_consec <= minDist_leaveReentry

            aoi_enterLeave_out(it_wl, 2)   = aoi_enterLeave_out(it_wl+1, 2);
            aoi_enterLeave_out(it_wl+1, :) = [];

        else

            it_wl = it_wl + 1;

        end
        clear offOn_consec

    end    

end