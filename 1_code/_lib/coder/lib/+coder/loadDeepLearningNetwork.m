function net = loadDeepLearningNetwork(matfile, varargin)

%   CODER.LOADDEEPLEARNINGNETWORK Load deep learning network, yolov2
%   object detector or an ssd object detector for code generation
%
%   NET = CODER.LOADDEEPLEARNINGNETWORK(FILENAME) loads a trained deep learning
%   SeriesNetwork, DAGNetwork, yolov2ObjectDetector or ssdObjectDetector
%   saved in a MAT-file. FILENAME must be a valid MAT-file existing in MATLAB path
%   containing a single SeriesNetwork, DAGNetwork, yolov2ObjectDetector or
%   ssdObjectDetector object.
%
%   NET = CODER.LOADDEEPLEARNINGNETWORK(FUNCTIONNAME) calls a function that
%   returns a trained SeriesNetwork, DAGNetwork, yolov2ObjectDetector or ssdObjectDetector
%   object. FUNCTIONNAME must be name of a function existing in
%   MATLAB path that returns a SeriesNetwork, DAGNetwork,
%   yolov2ObjectDetector or ssdObjectDetector object.
%
%   This function should be used when code is generated from a network
%   object for inference. This function generates a C++ class from this
%   network. The class name is derived from the MAT-file name, or the
%   function name.
%
%   NET = CODER.LOADDEEPLEARNINGNETWORK(FILENAME,NETWORK_NAME) is the same
%   as NET = CODER.LOADDEEPLEARNINGNETWORK(FILENAME) with the option to
%   name the C++ class generated from the network. NETWORK_NAME is a
%   descriptive name for the network object saved in the MAT-file, or the
%   function. It must be a char type that is a valid identifier in C++.
%
%   Example : Code generation from SeriesNetwork inference loaded from a
%   MAT-file
%
%   function out = alexnet_codegen(in)
%     %#codegen
%     persistent mynet;
%
%     if isempty(mynet)
%         mynet = coder.loadDeepLearningNetwork('imagenet-cnn.mat','alexnet');
%     end
%     out = mynet.predict(in);
%
%   Example : Code generation from SeriesNetwork inference loaded from a
%   function name. 'alexnet' is a Deep Learning Toolbox function that
%   returns a pretrained AlexNet model.
%
%   function out = alexnet_codegen(in)
%     %#codegen
%     persistent mynet;
%
%     if isempty(mynet)
%         mynet = coder.loadDeepLearningNetwork('alexnet','myAlexnet');
%     end
%     out = mynet.predict(in);
%
%   See also CNNCODEGEN

%   Copyright 2017-2019 The MathWorks, Inc.

%#codegen
narginchk(1, 2);
if coder.target('MATLAB')
    try
        net = coder.internal.loadDeepLearningNetwork(matfile, varargin{:});
        
    catch err
        throwAsCaller(err);
    end
    
else
    net = coder.internal.loadDeepLearningNetwork(matfile, varargin{:});   
end

end
