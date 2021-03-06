function [G_grad,sig_grad,pr_grad,dmu_dG_flat,dmu_dsig,dmu_dpr,det_zero] = compute_gradients_vb(params,mu_vb,stim,z_hidden)
%LEARNINGPARAMS.COMPUTE_GRADIENTS_VB compute gradient of log[q_vb] with
%respect to generative model params.
%
%Derivation uses implicit function theorem to find change in a fixed-point
%solution (mu_vb) with respect to the params of the distribution.

%dlogq_dmu = z_hidden(:) .* log(mu_vb(:)) + (1 - z_hidden(:)) .* log(1-mu_vb(:));
dlogq_dmu = z_hidden(:) ./(mu_vb(:)) - (1 - z_hidden(:)) ./(1-mu_vb(:));
det_zero = 0;

R = -params.G' * params.G;
H = params.Neurons_hidden;

% Apply the chain rule: d[log q]/dtheta = d[log q]/dmu * dmu/dtheta where
% dmu/dtheta uses the implicit function theorem:
%   dmu/dtheta = -inv(eye - dVB/dmu) * dF/dtheta)
dVB_dmu = repmat(mu_vb(:) .* (1-mu_vb(:)), [1 H]) .* (R - diag(diag(R))) / params.sigma_stim^2;
jacobian = (eye(H) - dVB_dmu);
if sum(sum(isnan(jacobian)))>0
    keyboard
end
if det(sym(jacobian))<=1e-5 || cond(jacobian) > 1e3
    G_grad = 0;
    sig_grad = 0;
    pr_grad = 0;
    dmu_dG_flat = 0;
    dmu_dsig = 0;
    dmu_dpr = 0;
    det_zero = 1;
else
    dmu_dG_flat = jacobian \ grad_vb_G_flat(params, mu_vb, stim)';
    dmu_dsig = jacobian \ grad_vb_sig(params, mu_vb, stim, R)';
    dmu_dpr = jacobian \ grad_vb_pr(params, mu_vb)';
    
    G_grad = reshape(dmu_dG_flat' * dlogq_dmu, [params.pix^2, params.Neurons_hidden]);
    sig_grad = dmu_dsig' * dlogq_dmu;
    pr_grad = dmu_dpr' * dlogq_dmu;
    
end

end

% % % % function dF_dG_flat = grad_vb_G_flat(params, mu_vb, stim)
% % % % % Gradient of VB update w.r.t. G (size |G| x neurons)
% % % %
% % % % flat_stim = reshape(stim, [params.pix^2, 1, 1]);
% % % % recurrent_terms = zeros(params.pix^2, params.Neurons_hidden, params.Neurons_hidden);
% % % % for h=1:params.Neurons_hidden
% % % %     mu = mu_vb(:);
% % % %     mu(h) = 1;
% % % %     recurrent_terms(:,:,h) = params.G .* mu';
% % % % end
% % % % dF_dG_flat = -mu_vb(:)' .* (1-mu_vb(:)') .* reshape(flat_stim + recurrent_terms, [params.pix^2*params.Neurons_hidden, params.Neurons_hidden]) / params.sigma_stim^2;
% % % %
% % % % end

% % function dF_dG_flat = grad_vb_G_flat(params, mu_vb, stim)
% % % Gradient of VB update w.r.t. G (size |G| x neurons)
% %
% % % feedforward term is stim * delta(i==k)
% % feedforward_term = kron(eye(params.Neurons_hidden), stim(:));
% %
% % % Recurrent term is PF when i=k or mu_i*PF_i otherwise
% % recurrent_term = zeros(size(feedforward_term));
% % for h=1:params.Neurons_hidden
% %     mu = mu_vb;
% %     mu(h) = 1;
% %     recurrent_term(:, h) = reshape(params.G .* mu(:)', numel(params.G), 1);
% % end
% %
% % dF_dG_flat = mu_vb(:)' .* (1 - mu_vb(:)') .* (feedforward_term - recurrent_term) / params.sigma_stim^2;
% % end

function dF_dG_flat = grad_vb_G_flat(params, mu_vb, stim)
% Gradient of VB update w.r.t. G (size |G| x neurons)

H = params.Neurons_hidden;
P = params.pix^2;

dF_dG = zeros(P, H, H);

for k=1:H
    for p=1:P
        for i=1:H
            if i == k
                ff = stim(p);
                rec = params.G(p,k);
            else
                ff = 0;
                rec = mu_vb(i) * params.G(p,k);
            end
            dF_dG(p,i,k) = mu_vb(k) * (1-mu_vb(k)) * (ff - rec) / params.sigma_stim^2;
        end
    end
end

dF_dG_flat = reshape(dF_dG, P*H, H);

end

function dF_dsig = grad_vb_sig(params, mu_vb, stim, R)
% Gradient of VB update w.r.t. sig (size 1 x neurons)

% TODO - WHY IS THIS CUBED?!?!
Rnotk = R - diag(diag(R));
dF_dsig = -mu_vb(:)' .* (1-mu_vb(:)') .* (2 * (stim(:)' * params.G + diag(R)' / 2 + (Rnotk * mu_vb(:))') / params.sigma_stim^3);
end

function dF_dpr = grad_vb_pr(params, mu_vb)
% Gradient of VB update w.r.t. pr  (size 1 x neurons)

dF_dpr = mu_vb(:)' .* (1-mu_vb(:)') ./ (params.prior * (1 - params.prior));
end