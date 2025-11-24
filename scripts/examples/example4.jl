using SketchySVD
using LinearAlgebra

# Simulate video: static background + moving foreground
n_frames = 200
height, width = 64, 64
m = height * width

# Background: low-rank
background = randn(height, width, 3)
background_flat = zeros(m, n_frames)
for t in 1:n_frames
    background_flat[:, t] = vec(background[:,:,1])  # Static
end

# Foreground: sparse
foreground = zeros(m, n_frames)
for t in 1:n_frames
    # Moving object
    obj_size = 10
    x_pos = 20 + div(t * width, n_frames)
    y_pos = 30
    indices = (y_pos:y_pos+obj_size-1, x_pos:min(x_pos+obj_size-1, width))
    
    for i in indices[1], j in indices[2]
        idx = (j-1) * height + i
        if idx <= m
            foreground[idx, t] = 100  # Bright object
        end
    end
end

# Composite video
video = background_flat + foreground

# Robust PCA via sketching
r_background = 5  # Background is low-rank
sketchy = init_sketchy(m=m, n=n_frames, r=r_background, ReduxMap=:sparse)

for t in 1:n_frames
    increment!(sketchy, video[:, t])
end

finalize!(sketchy)
U = sketchy.V
S = sketchy.Σ
V = sketchy.W
background_recovered = U * Diagonal(S) * V'
foreground_recovered = video - background_recovered

# Visualize results
using Plots
t_show = 100
plot(
    heatmap(reshape(video[:, t_show], height, width), title="Original Frame"),
    heatmap(reshape(background_recovered[:, t_show], height, width), title="Background"),
    heatmap(reshape(foreground_recovered[:, t_show], height, width), title="Foreground"),
    layout=(1,3), colorbar=false
)