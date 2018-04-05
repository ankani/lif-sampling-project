function pr = Proj_pr(pr)
mn = 0.01;
mx = 0.5;
if (pr <= mn)
    pr = mn;
else
    if(pr >= mx)
        pr = mx;
        
    end
end
end

