function [A,B] = moreinfo(msgid,varargin)
%CODER.INTERNAL.MOREINFO - Get a link to more information for a message ID.
%
%   Map-files contain topics that link to HTML files produced by doc
%   writers.  Doc writers add the topics as string to the XML from which
%   the HTML documentation is produced.
%
%   You can use the helpview command with a MAP file and a topic to display
%   the appropriate HTML.  Some of the HTML files are shipped as compressed
%   ZIP-files in JAR files.  I am not sure how this mechanism works.
%
%   MOREINFO(MSGID) returns a link to more information for msgID. Returns
%   '' if no additional information exists.  A link is a string containing
%   an appropriate <a href=...> ... </a>.  For testing purposes the magic
%   msgid 'Magic:Cookie:Link' will return a non-empty link.
%
%   MOREINFO('-topic',MSGID) returns the MAP-file topic for MSGID.
%
%   MOREINFO('-msgid',topic) returns the MSGID associated with topic if
%   any. Return '' if this TOPIC is syntactically invalid.  This reverse
%   lookup is useful for testing.
%
%   MOREINFO('-maps') return the cellarray of MAP files in which we look for
%   topics. (Needed to test that MAP files are valid.)
%
%   [MAPFILE, TOPIC] = MOREINFO('-lookup',MSGID) returns the name of the
%   MAPFILE and topic within that MAPFILE that was found.
%
%   MOREINFO('-open',MSGID) attempts to open the more-info topic associated
%   with the given MSGID if it exists and is resolvable.

%   Copyright 2010-2017 The MathWorks, Inc.

if ~usejava('jvm') || isdeployed()
    A = '';
    B = '';
    return;
end

switch msgid
    case  '-topic'
        assert(nargin == 2);
        A = msgid2topic(varargin{1});
    case '-msgid'
        assert(nargin == 2);
        A = topic2msgid(varargin{1});
    case '-lookup'
        assert(nargin == 2);
        [A,B] = lookup(varargin{1});
    case '-maps'
        assert(nargin==1);
        A = getMaps;
    case '-open'
        assert(nargin == 2);
        A = open(varargin{1});
    otherwise
        assert(nargin == 1);
        A = doit(msgid);
end

end

function topic = msgid2topic(msgid)
% This is a syntactic conversion before we do the lookup to see if the
% topic exists in any of our MAP files.
%
%  MAP-file topics may only contain the characters [a-zA-Z0-9_].  This
%  conversion tries to create a probabilistically unique bijection between
%  message IDs and topics.
%
%  Replace
%      : with _
%      _ with uUu
%
%  We use the quixotic uUu because it shouldn't show up in message ids by
%  itself, although nothing guarantees that.

topic = ['msginfo_' regexprep(msgid,{'_',':'},{'uUu','_'})];

end

function msgid = topic2msgid(topic)
% Must be the inverse of msgid2topic.
% This function does a purely syntactic check on purpose.  This is used for
% testing and should NOT check the existence of the msg id!
%
% See comments in msgid2topic for explanation of mapping.

X = regexp(topic,'msginfo_([\w_]+)','tokens');

if ~isscalar(X) || ~isscalar(X{1})
    msgid='';
else
    msgid = X{1}{1};
    msgid = regexprep(msgid,{'_','uUu'},{':','_'});
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function maps = getMaps
maps = { ...
    fullfile(docroot,'coder','helptargets.map') ...
    fullfile(docroot,'ecoder','helptargets.map') ...
    fullfile(docroot,'stateflow','stateflow.map') ...
    fullfile(docroot, 'simulink','helptargets.map') ...
    fullfile(docroot, 'fixedpoint', 'ug', 'fixedpoint_ug.map')
    };
end

function [mapfile,topic] = lookup(msgid)
if strcmp(msgid,magicMsgId)
    mapfile = 'MagicMap';
    topic = 'msginfo_Magic_Cookie_Link';
    return;
    
end
topic = msgid2topic(msgid);

persistent maps;
persistent cachedDocRoot;
% If docroot has been changed, repopulate the maps.
if isempty(maps) || ~strcmp(docroot, cachedDocRoot)
    cachedDocRoot = docroot;
    mapfiles = getMaps();
    maps = cell(size(mapfiles));
    for i = 1:numel(mapfiles)
        % Some MAP files are only available with certain products.
        if exist(mapfiles{i},'file')
            maps{i} = com.mathworks.mlwidgets.help.CSHelpTopicMap(mapfiles{i});
            assert(maps{i}.exists,'MAP file does not exist.');
        end
    end
    assert(~isempty(maps));
end

for i = 1:numel(maps)
    M = maps{i};
    if isempty(M)
        continue;
    end
    foundtopic = char(M.mapID(topic));
    if ~isempty(foundtopic)
        mapfile = char(M.getFilePath);
        return;
    end
end

mapfile = '';
topic = '';

end

function opened = open(msgid)
    opened = false;
    [mapfile, topic] = lookup(msgid);
    if ~isempty(mapfile) && ~isempty(topic)
        helpview(mapfile, topic);
        opened = true;
    end
end

function H = doit(msgid)

[mapfile,topic] = lookup(msgid);
if isempty(mapfile)
    H = '';
else
    msg = message('Coder:common:MoreInfo');
    
    % This use of help!view is sanctioned and okay.  It should not use
    % emlhelp.  We do this strange formatting of the string to prevent
    % test failures relating to using help!view in our MATLAB code.
    % (Normally not allowed).
    H = sprintf(['<a href="matlab:help' 'view(''%s'',''%s'');">%s</a>'], ...
        mapfile,topic,msg.getString());
end

end

function s  = magicMsgId()
  s = 'Magic:Cookie:Link';
end
% LocalWords:  MAPFILE
