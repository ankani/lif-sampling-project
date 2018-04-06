function z = enumerate(H)
%EM.ENUMERATE enumerate all values of H-dimensional z

assert(H <= 20, 'I refuse to enumerate over 2^20 states. Use truncation to limit this.');

z = zeros(2^H, H);

% The enumeration of all binary words along the rows of z, for example with H=3, looks like
%
%   0 0 0
%   0 0 1
%   0 1 0
%   0 1 1
%   1 0 0
%   1 0 1
%   1 1 0
%   1 1 1
%
% Notice that the columns follow a pattern of alternating blocks of [0s 1s], starting with half of
% the rows, then quarters, etc. Each column's 1s is 2 copies of a subsampled version of the previous
% column. We construct z in this column-wise way, where 'idx' 

halfsize = size(z, 1) / 2;
z(:, 1) = [zeros(halfsize, 1); ones(halfsize, 1)];

for h=2:H
    % Going from column h-1 to column h, keep every other value and duplicate them. 00001111 becomes
    % 00110011, which in turn becomes 01010101.
    z(:, h) = [z(1:2:end, h-1); z(1:2:end, h-1)];
end

end