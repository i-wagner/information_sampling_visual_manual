function [enabled] = getGpuEnabled(ctx)
    cfg = coder.gpu.getGpuConfig(ctx);
    enabled = false;
    if ~isempty(cfg)
        enabled = cfg.Enabled;
    end
end
