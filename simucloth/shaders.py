# Custom shader definition
mesh_vertex_shader = '''
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    in vec3 position;
    in vec3 normal;

    out vec3 v_normal;
    out vec3 v_position;
    out vec3 v_original_position;

    void main()
    {
        v_normal = normalize(mat3(modelViewMatrix) * normal);
        v_position = (modelViewMatrix * vec4(position, 1.0)).xyz;
        v_original_position = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
'''

mesh_fragment_shader = '''
    uniform vec3 lightDirection;
    uniform vec3 viewPosition;
    uniform vec3 color;
    uniform float shininess;
    uniform float fresnelBias;
    uniform float fresnelScale;
    uniform float fresnelPower;

    in vec3 v_normal;
    in vec3 v_position;
    in vec3 v_original_position;

    out vec4 fragColor;

    float ambient_occlusion(vec3 position, vec3 normal) {
        float ao = 0.0;
        float scale = 0.5;
        float offset = 0.0;
        for (int i = 0; i < 5; i++) {
            offset += scale;
            vec3 sample_pos = position + normal * offset;
            vec3 sample_vector = normalize(sample_pos - position);
            float sample_dist = length(sample_pos - position);
            ao += max(0.0, dot(normal, sample_vector)) / (1.0 + sample_dist);
        }
        return 1.0 - ao * 0.2;
    }

    void main()
    {
        vec3 N = normalize(v_normal);
        vec3 V = normalize(viewPosition - v_position);

        // Flip normal if it's a back-facing polygon
        if (dot(V, N) < 0.0) {
            N = -N;
        }

        vec3 L = normalize(lightDirection);
        vec3 H = normalize(L + V);

        float diffuse = max(dot(N, L), 0.0);
        float specular = pow(max(dot(N, H), 0.0), shininess);

        float fresnel = fresnelBias + fresnelScale * pow(1.0 - max(0.0, dot(N, V)), fresnelPower);

        float ao = ambient_occlusion(v_original_position, N);

        vec3 ambient = color * 0.1;
        vec3 diffuseColor = color * diffuse;
        vec3 specularColor = vec3(1.0) * specular;

        vec3 final_color = (ambient + diffuseColor + specularColor) * ao;
        final_color = mix(final_color, vec3(1.0), fresnel * 0.5);

        fragColor = vec4(final_color, 1.0);
    }
'''