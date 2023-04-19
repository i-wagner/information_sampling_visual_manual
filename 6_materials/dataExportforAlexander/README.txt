gazeShiftsForAlexander
2D matrix with gaze shifts that one participant made in the double-target condition of the visual search experiment; rows are gaze shifts, columns are different measures
(:,1): timestamp gaze shift ONSET (ms)
(:,2): timestamp gaze shift OFFSET (ms)
(:,3): mean HORIZONTAL gaze position between offset of this gaze shift and onset of the next one (dva)
(:,4): mean VERTICAL gaze position between offset of this gaze shift and onset of the next one (dva)
(:,5): index of fixated AOI (can be used to index to the stimulus coordinates in the coordinates variable); "666" means that background was fixated
(:,6): flag; easy target (1), hard target (2), easy distrator (3), hard distractor (4), or background (666) fixated
(:,7): # easy distractors in trial
(:,8): flag; easy target (1) or hard target chosen (2)
(:,9): # gaze shift after trial start (gaze shifts to the background are not counted for this)
(:,10): trial number

stimPosForAlexander
3D matrix with locations of all stimuli that were shown in a trial; rows are trials, columns are stimuli, pages are x- (:,:,1) and y-coordinates (:,:,2); entries of not-shown stimuli are "NaN"
(:,1): location easy target (dva)
(:,2): location difficult target (dva)
(:,3:10): locations of easy distractors (dva)
(:,11:18): locations of difficult distractors (dva)

Stimulus locations are relative to the position of the fixation cross, which was located at the lower screen half (coordinates [0 dva, -9.5 dva])