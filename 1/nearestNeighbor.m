
function [out] = nearestNeighbor(img, frac)
[m n] = size(img);
w = round(m*frac);
h = round(n*frac);
out = zeros(w, h);

for i = 1:w
    for j = 1:h
        out(i, j) = img(max(1, min(m, round(i/frac))), max(1, min(n, round(j/frac))));
    end
end

out = uint8(out);
end