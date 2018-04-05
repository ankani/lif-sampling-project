function sig = Proj_sig(sig)
mn = 0.5;
mx = 10;
if (sig <= mn)
    sig = mn;
else
    if (sig >= mx)
        sig = mx;
        
    end
end
end

	