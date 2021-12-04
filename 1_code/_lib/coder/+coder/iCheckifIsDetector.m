
function [isYOLO, isSSD] =  iCheckifIsDetector(matfile,variableName)
    coder.extrinsic('coder.loadMatObj');
    matObj = coder.loadMatObj(matfile,variableName);
    isYOLO = false;
    isSSD = false;
    if isa(matObj, 'yolov2ObjectDetector')
        isYOLO = true;
    elseif isa(matObj, 'ssdObjectDetector')
        isSSD=true;
    end
end
