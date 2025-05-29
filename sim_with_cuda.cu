#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simulation parameters
#define N 4096*4           
#define STEPS 1000        
#define DT 0.5f         // Smaller time step, more accurate the simulation... please note to adjust according to time as well.. smaller time step means lesser simulation as well
#define SOFTENING 1.0f    
#define SAVE_INTERVAL 10 

// Galaxy parameters
#define BLACK_HOLE_MASS 100000.0f  
#define GALAXY_SCALE 1000.0f       
#define NUM_GALAXIES 3            

#define G 1.0f 

typedef struct {
    float x, y, z;
} Vector3;

typedef struct {
    Vector3 position;
    Vector3 velocity;
    float mass;
    int type;      // 0 = star, 1 = black hole
    int galaxy_id;
} Body;

float check_minimum_distance(Body *bodies, int n) {
    float min_distance = FLT_MAX;
    
    // Check distances between all black holes
    for (int i = 0; i < NUM_GALAXIES; i++) {
        Body *bh1 = &bodies[i * ((n - NUM_GALAXIES) / NUM_GALAXIES + 1)];
        for (int j = i + 1; j < NUM_GALAXIES; j++) {
            Body *bh2 = &bodies[j * ((n - NUM_GALAXIES) / NUM_GALAXIES + 1)];
            
            float dx = bh2->position.x - bh1->position.x;
            float dy = bh2->position.y - bh1->position.y;
            float dz = bh2->position.z - bh1->position.z;
            
            float distance = sqrt(dx*dx + dy*dy + dz*dz);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
    }
    
    return min_distance;
}

// CUDA kernel for updating positions and velocities using Leapfrog integrator
__global__ void compute_gravity(Body *bodies, int n, float dt, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    Vector3 pos_i = bodies[i].position;
    Vector3 acc = {0.0f, 0.0f, 0.0f};
    
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        
        float dx = bodies[j].position.x - pos_i.x;
        float dy = bodies[j].position.y - pos_i.y;
        float dz = bodies[j].position.z - pos_i.z;
        
        float distSqr = dx*dx + dy*dy + dz*dz + softening;
        
        // Compute gravitational force (Newton's law)
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        
        float force = G * bodies[j].mass * invDist3;
        
        float max_force = 100.0f;
        force = (force > max_force) ? max_force : force;
        
        acc.x += dx * force;
        acc.y += dy * force;
        acc.z += dz * force;
    }
    
    bodies[i].velocity.x += acc.x * dt;
    bodies[i].velocity.y += acc.y * dt;
    bodies[i].velocity.z += acc.z * dt;
    
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
    bodies[i].position.z += bodies[i].velocity.z * dt;
}

float random_float(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

void create_galaxy(Body *bodies, int start_idx, int count, Vector3 center, Vector3 velocity, 
                  float scale, float black_hole_mass, int galaxy_id, float spin_factor) {

    // Place the central black hole
    bodies[start_idx].position = center;
    bodies[start_idx].velocity = velocity;
    bodies[start_idx].mass = black_hole_mass;
    bodies[start_idx].type = 1;
    bodies[start_idx].galaxy_id = galaxy_id;
    
    float innerRadius = scale * 0.1f;
    float outerRadius = scale;
    
    for (int i = 1; i < count; i++) {
        int idx = start_idx + i;
        
        float theta = 2.0f * M_PI * ((float)rand() / RAND_MAX);

        float r = innerRadius + (outerRadius - innerRadius) * sqrt((float)rand() / RAND_MAX);
        
        float arm_phase = 4.0f; // Number of spiral arms effect
        theta += arm_phase * log(r / innerRadius);
        
        bodies[idx].position.x = center.x + r * cos(theta);
        bodies[idx].position.y = center.y + r * sin(theta);
        
        // Z position - thinner near center, thicker at edges
        float z_scale = 0.1f * scale * (0.1f + 0.9f * r / outerRadius);
        bodies[idx].position.z = center.z + z_scale * ((float)rand() / RAND_MAX - 0.5f);
        
        // Calculate circular orbital velocity for stable orbits
        float orbit_speed = sqrt(G * black_hole_mass / r);
        
        // Set velocity for a stable orbit perpendicular to radius vector
        bodies[idx].velocity.x = velocity.x - orbit_speed * sin(theta) * spin_factor;
        bodies[idx].velocity.y = velocity.y + orbit_speed * cos(theta) * spin_factor;
        bodies[idx].velocity.z = velocity.z + ((float)rand() / RAND_MAX - 0.5f) * 0.1f * orbit_speed;
        
        bodies[idx].mass = 0.1f + ((float)rand() / RAND_MAX) * 2.0f;
        bodies[idx].type = 0;
        bodies[idx].galaxy_id = galaxy_id;
    }
}

void initialize_bodies(Body *bodies, int n) {
    int stars_per_galaxy = (n - NUM_GALAXIES) / NUM_GALAXIES;
    
    for (int g = 0; g < NUM_GALAXIES; g++) {
        // Calculate position in a spherical distribution around origin
        float angle = 2.0f * M_PI * g / NUM_GALAXIES;
        float radius = 2.5f * GALAXY_SCALE;
        float height = radius * 0.2f * (g % 2 == 0 ? 1 : -1);  // Alternate above and below plane
        
        Vector3 center = {
            radius * cos(angle),
            radius * sin(angle),
            height
        };
        
        float speed = sqrt(G * BLACK_HOLE_MASS * NUM_GALAXIES / radius) * (0.8f + 0.4f * random_float(0, 1));
        Vector3 velocity = {
            -speed * sin(angle) * (g % 2 == 0 ? 1 : -1),
            speed * cos(angle) * (g % 2 == 0 ? 1 : -1),
            speed * 0.1f * (random_float(0, 1) - 0.5f)
        };
        
        // Random for fun hehe
        float galaxy_scale = GALAXY_SCALE * (0.8f + 0.4f * random_float(0, 1));
        float galaxy_mass = BLACK_HOLE_MASS * (0.8f + 0.4f * random_float(0, 1));
        float spin_factor = (g % 2 == 0 ? 1.0f : -1.0f) * (0.8f + 0.4f * random_float(0, 1));
        
        create_galaxy(bodies, 
                     g * (stars_per_galaxy + 1), 
                     stars_per_galaxy + 1, 
                     center, 
                     velocity, 
                     galaxy_scale, 
                     galaxy_mass, 
                     g, 
                     spin_factor);
        
        printf("Galaxy %d: Position (%.1f, %.1f, %.1f), Velocity (%.1f, %.1f, %.1f)\n",
               g, center.x, center.y, center.z, velocity.x, velocity.y, velocity.z);
    }
}

float check_galaxy_interaction(Body *bodies) {
    return check_minimum_distance(bodies, N);
}

void save_positions(Body *bodies, int n, int step, float galaxy_distance) {
    char filename[256];
    sprintf(filename, "positions_%04d.csv", step);
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "# step=%d distance=%.2f\n", step, galaxy_distance);
    fprintf(f, "x,y,z,vx,vy,vz,mass,type,galaxy\n");
    
    for (int i = 0; i < n; i++) {
        fprintf(f, "%f,%f,%f,%f,%f,%f,%f,%d,%d\n",
                bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
                bodies[i].velocity.x, bodies[i].velocity.y, bodies[i].velocity.z,
                bodies[i].mass, bodies[i].type, bodies[i].galaxy_id);
    }
    
    fclose(f);
}

void generate_visualization_script() {
    FILE *f = fopen("visualize_galaxy.py", "w");
    if (!f) {
        printf("Error creating visualization script\n");
        return;
    }
    
    fprintf(f, "import pandas as pd\n");
    fprintf(f, "import numpy as np\n");
    fprintf(f, "import matplotlib.pyplot as plt\n");
    fprintf(f, "from matplotlib.animation import FuncAnimation\n");
    fprintf(f, "from matplotlib import colors\n");
    fprintf(f, "from mpl_toolkits.mplot3d import Axes3D\n");
    fprintf(f, "from matplotlib.colors import ListedColormap\n");
    fprintf(f, "import colorsys\n");
    fprintf(f, "import os\n\n");
    
    fprintf(f, "def read_frame(frame):\n");
    fprintf(f, "    try:\n");
    fprintf(f, "        filename = 'positions_{:04d}.csv'.format(frame * %d)\n", SAVE_INTERVAL);
    fprintf(f, "        if not os.path.exists(filename):\n");
    fprintf(f, "            return None, None\n");
    fprintf(f, "            \n");
    fprintf(f, "        # First line contains metadata\n");
    fprintf(f, "        with open(filename, 'r') as f:\n");
    fprintf(f, "            metadata_line = f.readline().strip('#').strip()\n");
    fprintf(f, "        \n");
    fprintf(f, "        meta_dict = {}\n");
    fprintf(f, "        for item in metadata_line.split():\n");
    fprintf(f, "            if '=' in item:\n");
    fprintf(f, "                key, value = item.split('=')\n");
    fprintf(f, "                try:\n");
    fprintf(f, "                    meta_dict[key] = float(value) if '.' in value else int(value)\n");
    fprintf(f, "                except ValueError:\n");
    fprintf(f, "                    meta_dict[key] = value\n");
    fprintf(f, "        \n");
    fprintf(f, "        data = pd.read_csv(filename, comment='#')\n");
    fprintf(f, "        return data, meta_dict\n");
    fprintf(f, "    except Exception as e:\n");
    fprintf(f, "        print(f\"Error reading frame {frame}: {str(e)}\")\n");
    fprintf(f, "        return None, None\n\n");
    
    fprintf(f, "fig = plt.figure(figsize=(18, 10))\n");
    fprintf(f, "ax1 = fig.add_subplot(121, projection='3d')\n");
    fprintf(f, "ax2 = fig.add_subplot(122)\n\n");
    
    fprintf(f, "# Fixed view limits for the top-down view\n");
    fprintf(f, "FIXED_VIEW_LIMIT = 3000  # Adjust based on your galaxy scale\n\n");
    
    fprintf(f, "def get_distinct_colors(n):\n");
    fprintf(f, "    colors = []\n");
    fprintf(f, "    for i in range(n):\n");
    fprintf(f, "        hue = i / float(n)\n");
    fprintf(f, "        lightness = 0.5 + 0.2 * ((i %% 3) / 3.0)\n");
    fprintf(f, "        saturation = 0.7 + 0.2 * ((i %% 5) / 5.0)\n");
    fprintf(f, "        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)\n");
    fprintf(f, "        colors.append(rgb)\n");
    fprintf(f, "    return colors\n\n");
    
    fprintf(f, "galaxy_colors = get_distinct_colors(%d)\n", NUM_GALAXIES);
    fprintf(f, "black_hole_colors = [(c[0]*0.8, c[1]*0.8, c[2]*0.8) for c in galaxy_colors]\n\n");
    
    fprintf(f, "# Track galaxy distances over time\n");
    fprintf(f, "distances = []\n");
    fprintf(f, "steps = []\n\n");
    
    fprintf(f, "interaction_cmap = plt.cm.cool\n\n");
    
    fprintf(f, "def update(frame):\n");
    fprintf(f, "    ax1.clear()\n");
    fprintf(f, "    ax2.clear()\n");
    fprintf(f, "    \n");
    fprintf(f, "    data, metadata = read_frame(frame)\n");
    fprintf(f, "    if data is None:\n");
    fprintf(f, "        return\n");
    fprintf(f, "    \n");
    fprintf(f, "    if metadata and 'distance' in metadata:\n");
    fprintf(f, "        current_distance = metadata['distance']\n");
    fprintf(f, "        distances.append(current_distance)\n");
    fprintf(f, "        steps.append(metadata['step'])\n");
    fprintf(f, "    else:\n");
    fprintf(f, "        current_distance = 0\n");
    fprintf(f, "    \n");
    fprintf(f, "    black_holes = data[data['type'] == 1]\n");
    fprintf(f, "    stars = data[data['type'] == 0]\n");
    fprintf(f, "    \n");
    fprintf(f, "    # Filter out particles beyond fixed view limits for 2D plot\n");
    fprintf(f, "    stars_2d = stars[\n");
    fprintf(f, "        (stars['x'] >= -FIXED_VIEW_LIMIT) & \n");
    fprintf(f, "        (stars['x'] <= FIXED_VIEW_LIMIT) & \n");
    fprintf(f, "        (stars['y'] >= -FIXED_VIEW_LIMIT) & \n");
    fprintf(f, "        (stars['y'] <= FIXED_VIEW_LIMIT)\n");
    fprintf(f, "    ]\n");
    fprintf(f, "    \n");
    fprintf(f, "    # Calculate interaction strength for color variation\n");
    fprintf(f, "    interaction_strength = max(0, min(1, 2000 / (current_distance + 1)))\n");
    fprintf(f, "    size_factor = 1.0 + interaction_strength * 0.5\n");
    fprintf(f, "    \n");
    fprintf(f, "    max_range = max(abs(data['x'].max()), abs(data['y'].max()), \n");
    fprintf(f, "                   abs(data['x'].min()), abs(data['y'].min())) * 1.1\n");
    fprintf(f, "    z_range = max(abs(data['z'].max()), abs(data['z'].min())) * 1.5\n");
    fprintf(f, "    \n");
    fprintf(f, "    # 3D Plot - dynamic view\n");
    fprintf(f, "    ax1.set_xlim(-max_range, max_range)\n");
    fprintf(f, "    ax1.set_ylim(-max_range, max_range)\n");
    fprintf(f, "    ax1.set_zlim(-z_range, z_range)\n");
    fprintf(f, "    \n");
    fprintf(f, "    # 2D Plot - fixed view\n");
    fprintf(f, "    ax2.set_xlim(-FIXED_VIEW_LIMIT, FIXED_VIEW_LIMIT)\n");
    fprintf(f, "    ax2.set_ylim(-FIXED_VIEW_LIMIT, FIXED_VIEW_LIMIT)\n");
    fprintf(f, "    \n");
    fprintf(f, "    # Plot stars for each galaxy in 3D view (all stars)\n");
    fprintf(f, "    for gid in range(%d):\n", NUM_GALAXIES);
    fprintf(f, "        g_stars = stars[stars['galaxy'] == gid]\n");
    fprintf(f, "        \n");
    fprintf(f, "        if len(g_stars) > 0:\n");
    fprintf(f, "            sizes = np.minimum(g_stars['mass'] * size_factor, 3.0 * size_factor)\n");
    fprintf(f, "            \n");
    fprintf(f, "            velocities = np.sqrt(g_stars['vx']**2 + g_stars['vy']**2 + g_stars['vz']**2)\n");
    fprintf(f, "            if len(velocities) > 0:\n");
    fprintf(f, "                v_min, v_max = velocities.min(), velocities.max()\n");
    fprintf(f, "                if v_max > v_min:\n");
    fprintf(f, "                    v_norm = (velocities - v_min) / (v_max - v_min)\n");
    fprintf(f, "                else:\n");
    fprintf(f, "                    v_norm = np.zeros_like(velocities)\n");
    fprintf(f, "            else:\n");
    fprintf(f, "                v_norm = np.array([])\n");
    fprintf(f, "            \n");
    fprintf(f, "            base_color = colors.to_rgba(galaxy_colors[gid])\n");
    fprintf(f, "            \n");
    fprintf(f, "            # Blend with velocity-based color when galaxies are interacting\n");
    fprintf(f, "            if len(v_norm) > 0:\n");
    fprintf(f, "                custom_colors = []\n");
    fprintf(f, "                for v in v_norm:\n");
    fprintf(f, "                    vel_color = colors.to_rgba(interaction_cmap(v))\n");
    fprintf(f, "                    blended = tuple(base_color[i] * (1-interaction_strength) + \n");
    fprintf(f, "                                   vel_color[i] * interaction_strength for i in range(3)) + (0.7,)\n");
    fprintf(f, "                    custom_colors.append(blended)\n");
    fprintf(f, "                \n");
    fprintf(f, "                ax1.scatter(g_stars['x'], g_stars['y'], g_stars['z'], \n");
    fprintf(f, "                           s=sizes, c=custom_colors, alpha=0.6)\n");
    fprintf(f, "            else:\n");
    fprintf(f, "                ax1.scatter(g_stars['x'], g_stars['y'], g_stars['z'], \n");
    fprintf(f, "                           s=sizes, color=galaxy_colors[gid], alpha=0.6)\n");
    fprintf(f, "    \n");
    fprintf(f, "    # Plot stars for each galaxy in 2D view (only stars within view limits)\n");
    fprintf(f, "    for gid in range(%d):\n", NUM_GALAXIES);
    fprintf(f, "        g_stars_2d = stars_2d[stars_2d['galaxy'] == gid]\n");
    fprintf(f, "        \n");
    fprintf(f, "        if len(g_stars_2d) > 0:\n");
    fprintf(f, "            sizes = np.minimum(g_stars_2d['mass'] * size_factor, 3.0 * size_factor)\n");
    fprintf(f, "            \n");
    fprintf(f, "            velocities = np.sqrt(g_stars_2d['vx']**2 + g_stars_2d['vy']**2 + g_stars_2d['vz']**2)\n");
    fprintf(f, "            if len(velocities) > 0:\n");
    fprintf(f, "                v_min, v_max = velocities.min(), velocities.max()\n");
    fprintf(f, "                if v_max > v_min:\n");
    fprintf(f, "                    v_norm = (velocities - v_min) / (v_max - v_min)\n");
    fprintf(f, "                else:\n");
    fprintf(f, "                    v_norm = np.zeros_like(velocities)\n");
    fprintf(f, "            else:\n");
    fprintf(f, "                v_norm = np.array([])\n");
    fprintf(f, "            \n");
    fprintf(f, "            base_color = colors.to_rgba(galaxy_colors[gid])\n");
    fprintf(f, "            \n");
    fprintf(f, "            if len(v_norm) > 0:\n");
    fprintf(f, "                custom_colors = []\n");
    fprintf(f, "                for v in v_norm:\n");
    fprintf(f, "                    vel_color = colors.to_rgba(interaction_cmap(v))\n");
    fprintf(f, "                    blended = tuple(base_color[i] * (1-interaction_strength) + \n");
    fprintf(f, "                               vel_color[i] * interaction_strength for i in range(3)) + (0.7,)\n");
    fprintf(f, "                    custom_colors.append(blended)\n");
    fprintf(f, "                \n");
    fprintf(f, "                ax2.scatter(g_stars_2d['x'], g_stars_2d['y'], \n");
    fprintf(f, "                           s=sizes, c=custom_colors, alpha=0.6)\n");
    fprintf(f, "            else:\n");
    fprintf(f, "                ax2.scatter(g_stars_2d['x'], g_stars_2d['y'], \n");
    fprintf(f, "                           s=sizes, color=galaxy_colors[gid], alpha=0.6)\n");
    fprintf(f, "    \n");
    fprintf(f, "    for i, bh in black_holes.iterrows():\n");
    fprintf(f, "        gid = int(bh['galaxy'])\n");
    fprintf(f, "        \n");
    fprintf(f, "        ax1.scatter([bh['x']], [bh['y']], [bh['z']], \n");
    fprintf(f, "                   s=40, color='yellow', edgecolor=galaxy_colors[gid], linewidth=1)\n");
    fprintf(f, "                   \n");
    fprintf(f, "        # Plot black hole in 2D view only if within limits\n");
    fprintf(f, "        if abs(bh['x']) <= FIXED_VIEW_LIMIT and abs(bh['y']) <= FIXED_VIEW_LIMIT:\n");
    fprintf(f, "            ax2.scatter([bh['x']], [bh['y']], \n");
    fprintf(f, "                       s=70, color='yellow', edgecolor=galaxy_colors[gid], linewidth=1)\n");
    fprintf(f, "    \n");
    fprintf(f, "    if len(black_holes) > 1 and interaction_strength > 0.3:\n");
    fprintf(f, "        bh_xs = black_holes['x'].values\n");
    fprintf(f, "        bh_ys = black_holes['y'].values\n");
    fprintf(f, "        bh_zs = black_holes['z'].values\n");
    fprintf(f, "        \n");
    fprintf(f, "        for i in range(len(black_holes)):\n");
    fprintf(f, "            for j in range(i+1, len(black_holes)):\n");
    fprintf(f, "                ax1.plot([bh_xs[i], bh_xs[j]], [bh_ys[i], bh_ys[j]], [bh_zs[i], bh_zs[j]], \n");
    fprintf(f, "                         'y--', alpha=0.4 * interaction_strength)\n");
    fprintf(f, "                \n");
    fprintf(f, "                if (abs(bh_xs[i]) <= FIXED_VIEW_LIMIT and abs(bh_ys[i]) <= FIXED_VIEW_LIMIT) or \\\n");
    fprintf(f, "                   (abs(bh_xs[j]) <= FIXED_VIEW_LIMIT and abs(bh_ys[j]) <= FIXED_VIEW_LIMIT):\n");
    fprintf(f, "                    ax2.plot([bh_xs[i], bh_xs[j]], [bh_ys[i], bh_ys[j]], 'y--', \n");
    fprintf(f, "                             alpha=0.4 * interaction_strength)\n");
    fprintf(f, "    \n");
    fprintf(f, "    if len(distances) > 1:\n");
    fprintf(f, "        inset_ax = ax2.inset_axes([0.65, 0.65, 0.3, 0.3])\n");
    fprintf(f, "        inset_ax.plot(steps, distances, 'k-')\n");
    fprintf(f, "        inset_ax.scatter([metadata['step']], [current_distance], color='red')\n");
    fprintf(f, "        inset_ax.set_title('Min Galaxy Distance')\n");
    fprintf(f, "        inset_ax.grid(True, alpha=0.3)\n");
    fprintf(f, "    \n");
    fprintf(f, "    ax1.set_xlabel('X')\n");
    fprintf(f, "    ax1.set_ylabel('Y')\n");
    fprintf(f, "    ax1.set_zlabel('Z')\n");
    fprintf(f, "    ax1.set_title('3D View')\n");
    fprintf(f, "    \n");
    fprintf(f, "    ax2.set_xlabel('X')\n");
    fprintf(f, "    ax2.set_ylabel('Y')\n");
    fprintf(f, "    ax2.set_title('Top-Down View (Fixed Scale)')\n");
    fprintf(f, "    ax2.set_aspect('equal')\n");
    fprintf(f, "    \n");
    fprintf(f, "    visible_count = len(stars_2d)\n");
    fprintf(f, "    total_count = len(stars)\n");
    fprintf(f, "    outside_count = total_count - visible_count\n");
    fprintf(f, "    \n");
    fprintf(f, "    plt.suptitle(f'Galaxy Collision Simulation - Step {metadata[\"step\"]}\\n'\n");
    fprintf(f, "                f'Min distance between galaxies: {current_distance:.1f} units\\n'\n");
    fprintf(f, "                f'Visible particles: {visible_count} (invisible: {outside_count})')\n");
    fprintf(f, "    return fig\n\n");
    
    fprintf(f, "print(\"Creating animation...\")\n");
    fprintf(f, "anim = FuncAnimation(fig, update, frames=range(%d), interval=100)\n", STEPS/SAVE_INTERVAL + 1);
    fprintf(f, "print(\"Saving animation...\")\n");
    fprintf(f, "anim.save('galaxy_collision.mp4', writer='ffmpeg', fps=15, dpi=150, extra_args=['-vcodec', 'libx264'])\n");
    fprintf(f, "print(\"Animation saved as galaxy_collision.mp4\")\n");
    fprintf(f, "plt.show()\n");
    
    fclose(f);
    printf("Visualization script created: visualize_galaxy.py\n");
}

int main() {
    printf("Starting Galaxy Collision Simulation with %d particles and %d galaxies\n", N, NUM_GALAXIES);
    
    // Allocate host memory
    Body *h_bodies = (Body*)malloc(N * sizeof(Body));
    
    srand(42);
    initialize_bodies(h_bodies, N);
    printf("Galaxies initialized\n");
    
    // Allocate device memory
    Body *d_bodies;
    cudaMalloc(&d_bodies, N * sizeof(Body));
    
    // Copy data to device
    cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
    
    // Calculate GPU execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float galaxy_distance = check_galaxy_interaction(h_bodies);
    printf("Initial minimum distance between galaxies: %.2f\n", galaxy_distance);
    
    save_positions(h_bodies, N, 0, galaxy_distance);
    printf("Initial state saved\n");

    float base_dt = DT;
    float current_dt = base_dt;
    
    // Main simulation loop
    for (int step = 1; step <= STEPS; step++) {
        clock_t start = clock();
        compute_gravity<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, N, current_dt, SOFTENING);
        clock_t end = clock();
        double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        // Wait for GPU to finish
        cudaDeviceSynchronize();
        
        if (step % SAVE_INTERVAL == 0) {
            // Copy data back to host
            cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

            float min_distance = check_minimum_distance(h_bodies, N);
    
            // Calculate adaptive time step
            current_dt = base_dt * fmin(1.0f, min_distance / (2.0f * GALAXY_SCALE));
            current_dt = fmax(current_dt, base_dt * 0.01f);
            
            galaxy_distance = check_galaxy_interaction(h_bodies);            
            save_positions(h_bodies, N, step, galaxy_distance);            
            printf("Step %d/%d completed (%.1f%%) - Min galaxy distance: %.2f - CPU Time: %.4f sec\n", 
                  step, STEPS, 100.0f * step / STEPS, galaxy_distance, cpu_time);
        }
    }
    
    generate_visualization_script();    
    cudaFree(d_bodies);
    free(h_bodies);
    
    printf("Simulation complete!\n");
    printf("To visualize results, run: python visualize_galaxy.py\n");
    
    return 0;
}