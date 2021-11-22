#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

float pi = 3.14159265359;
float degToRad = 3.14159265359 / 180.0;

float animSpeed = 0.01;

uniform int[10] u_Word;

struct MapQuery
{
  float dist;
  float displacement;
  float material;
};

struct Material
{
    vec3 color;
    float kd;
    float ks;
    float cosPow;
    float displacement;
};

struct PointLight
{
    vec3 position;
    vec3 color;
    bool castsShadow;
    float ambient;
};

float hash3(vec3 v)
{
    return fract(sin(dot(v, vec3(24.51853, 4815.44774, 32555.33333))) * 3942185.3);
}

vec4 noise3(vec3 v)
{
    //Adapted from IQ: https://www.iquilezles.org/www/articles/morenoise/morenoise.htm
    vec3 intV = floor(v);
    vec3 fractV = fract(v);
    vec3 u = fractV*fractV*fractV*(fractV*(fractV*6.0-15.0)+10.0);
    vec3 du = 30.0*fractV*fractV*(fractV*(fractV-2.0)+1.0);
    
    float a = hash3( intV+vec3(0.f,0.f,0.f) );
    float b = hash3( intV+vec3(1.f,0.f,0.f) );
    float c = hash3( intV+vec3(0.f,1.f,0.f) );
    float d = hash3( intV+vec3(1.f,1.f,0.f) );
    float e = hash3( intV+vec3(0.f,0.f,1.f) );
    float f = hash3( intV+vec3(1.f,0.f,1.f) );
    float g = hash3( intV+vec3(0.f,1.f,1.f) );
    float h = hash3( intV+vec3(1.f,1.f,1.f) );
    
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;
    
    
    vec3 dv = 2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                             k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                             k3 + k6*u.x + k5*u.y + k7*u.x*u.y);
    
    return vec4(-1.f+2.f*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), dv);
}

vec4 fbm3(vec3 v, int octaves, float amp, float freq, float pers, float freq_power)
{
    float sum = 0.f;
    vec3 dv = vec3(0.f,0.f,0.f);
    float speed = 0.01f;
    for(int i = 0; i < octaves; ++i)
    {
        amp *= pers;
        freq *= freq_power;
        vec4 noise = noise3((v) * freq);
        sum += amp * noise.x;
        dv += amp * noise.yzw;
    }
    return vec4(sum, dv);
}

MapQuery smoothMin( MapQuery a, MapQuery b, float k)
{
    float h = clamp( 0.5+0.5*(b.dist-a.dist)/k, 0.0, 1.0 );
    MapQuery res;
    if(h < 0.5) {
        res.dist = mix( b.dist, a.dist, h ) - k*h*(1.0-h);
        res.material = b.material;
    } else {
        res.dist = mix( b.dist, a.dist, h ) - k*h*(1.0-h);
        res.material = a.material;
    }
    return res;
}

vec2 smin( vec2 a, vec2 b, float k )
{
    float h = clamp( 0.5+0.5*(b.x-a.x)/k, 0.0, 1.0 );
    if(h < 0.5) {
        return vec2(mix( b.x, a.x, h ) - k*h*(1.0-h), b.y);
    } else {
        return vec2(mix( b.x, a.x, h ) - k*h*(1.0-h), a.y);
    }
}

vec2 smax( vec2 a, vec2 b, float k )
{
    float h = max(k-abs(a.x-b.x),0.0);
    return (a.x > b.x ? vec2((a.x + h*h*0.25/k), a.y) : vec2((b.x + h*h*0.25/k), b.y));
}


vec3 elongate( vec3 p, vec3 h )
{
    vec3 q = p - clamp( p, -h, h );
    return q;
}
float cappedCone(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;
    float x = sqrt( papa - paba*paba*baba );
    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;
    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );
    float cbx = x-ra - f*rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

float roundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;
    
    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot( pa*l2 - ba*y, pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

// Ra: radius rb: roundedness h: height
float roundedCylinder( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float cappedTorus(vec3 p, vec2 sc, float ra, float rb)
{
  p.x = abs(p.x);
  float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
  return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

float box( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}


float sphere(vec3 p, float s)
{
    return length(p) - s;
}

float ellipsoid( vec3 p, vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float stick(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    vec3 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return  length( pa - ba*h ) - mix(r1,r2,h*h*(3.0-2.0*h));
}


vec3 bend( vec3 p, float k )
{
    float c = cos(k*p.x);
    float s = sin(k*p.x);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xy,p.z);
    return q;
}


vec3 twist( vec3 p, float k)
{
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return q;
}

float onion(float sdf, float thickness)
{
    return abs(sdf) - thickness;
}

float roundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

// https://iquilezles.org/www/articles/noacos/noacos.htm
mat3 rotationAxisAngle( vec3 v, float a )
{
    float si = sin( a );
    float co = cos( a );
    float ic = 1.0f - co;

    return mat3( v.x*v.x*ic + co,       v.y*v.x*ic - si*v.z,    v.z*v.x*ic + si*v.y,
                   v.x*v.y*ic + si*v.z,   v.y*v.y*ic + co,        v.z*v.y*ic - si*v.x,
                   v.x*v.z*ic - si*v.y,   v.y*v.z*ic + si*v.x,    v.z*v.z*ic + co );
}

float plane( vec3 p, vec3 n, float h )
{
  // n must be normalized
  return dot(p,n) + h;
}

float getBias(float time, float bias)
{
    return (time / ((((1.0/bias) - 2.0)*(1.0 - time))+1.0));
}

float getGain(float time,float gain)
{
  if(time < 0.5)
    return getBias(time * 2.0,gain)/2.0;
  else
    return getBias(time * 2.0 - 1.0,1.0 - gain)/2.0 + 0.5;
}


vec2 map(vec3 p)
{
    
    float modTime = mod(u_Time * animSpeed, 10000.0);
    float sinTime = 0.5 * (sin(0.002 * modTime) + 1.0);
    float amp = 0.3;
    float freq = 0.6;
    vec3 offset = vec3(0.0);
    
    for(int i = 0; i < 8; ++i) {
        if(u_Word[i] > 0) {
            int index = u_Word[i];

            int x = index / (9) + 1;
            int y = (index / 3) % 3 + 1;
            int z = index % 3 + 1;
            vec3 letter = vec3(float(x), float(y), float(z));
            freq *= 1.7;
            
            freq = min(freq, 24.0);
            amp *= 0.54;
            offset += amp * (sin(p.x * freq + letter.x) + sin(p.y * freq + letter.y) + sin(p.z * freq + letter.z));
        }
    }
    
   // offset *= sinTime;
    vec3 distorted = p + offset.yzx;
    vec2 face = vec2(sphere(distorted, 1.2), 0.0);
    /*
    for(int i = 0; i < 8; ++i) {
        if(u_Word[i] > 0) {
            int index = u_Word[i];
            int x = index / (9) + 1;
            int y = (index / 3) % 3 + 1;
            int z = index % 3 + 1;
            vec3 letter = 0.1 * vec3(float(x), float(y), float(z));
            face = smin(face, vec2(sphere(distorted + letter.zxy, 0.3), 0.0), 0.2);
        }
    }*/
    return face;
}

int numIterations = 40;
vec4 getAlbedo(vec3 p, float dist) {
    
    float floorTime = floor(mod(u_Time * animSpeed, 26.0));
    float modTime = mod(u_Time * animSpeed, 1000.0);
    float amp = 0.8;
    float freq = 1.2;
    vec3 albedo = vec3(0.0);
    
    float alpha = 0.0;
    float wordLength = 0.0;
    
    vec3 wordsum = vec3(0.0);

    vec4 noise = vec4(0.0);
    float noiseAmp = 1.1;
    float noiseFreq = 1.2;

    for(int i = 0; i < 8; ++i) {
        if(u_Word[i] > 0) {
            int index = u_Word[i] % 26 + 1;
            ++wordLength;
            int x = index / (9) + 1;
            int y = (index / 3) % 3 + 1;
            int z = index % 3 + 1;
            vec3 letter = 0.3 * vec3(float(x), float(y), float(z));
            freq *= 1.9;
            freq = min(freq, 10.0);
            amp *= 0.9;
            
            noiseAmp *= 0.5;
            noiseFreq *= 2.0;
            vec3 offP = p.xyz + normalize(letter) * modTime;

            vec4 noiseOctave = noiseAmp * noise3(noiseFreq * (p + letter));

            noise += noiseOctave;
            
            
            wordsum += letter;
            
            offP = p.xyz + letter;
            float a = abs(amp * (sin(freq * offP.x) + sin(freq * offP.y) + sin(freq * offP.z)));
            
            if(dist > -0.1) {
                alpha += a * noiseOctave.x;

            } else {
                alpha += a * 3.0;
            }
            albedo += letter * a;

            /*
            alpha += a; //abs(noise.x);
            albedo += letter * a;*/
        }
    }
    // noise = fbm3(p + wordsum, 5, 1.1, 1.0, 0.5, 1.8);
    alpha *= noise.x;
    //comment in for trig implementation
    alpha /= 10.0 * wordLength * float(numIterations);

    //comment in for fbm implementation
    //alpha /= 50.0 * wordLength * float(numIterations);
    alpha = clamp(alpha, 0.0, 1.0);
    albedo = normalize(albedo);

    return vec4(albedo, alpha);
}


vec4 accumulateDensity(vec3 p) {
    float t = 0.0;
    vec4 col = vec4(0.0);
    vec3 rayDir = normalize(p - u_Eye.xyz);
    for(int i = 0; i < numIterations; ++i)
    {
        vec3 pos = p.xyz + rayDir.xyz * t;
        vec4 density = vec4(0.0,0.0,0.0,0.015);
        
        //Comment in for cloud
        float dist = map(pos).x;
        if (dist < 0.01) {
            density = getAlbedo(pos, dist);
            col += density;
            t += max(t * 0.001 + density.a * 0.1, 0.015);
            
        } else {
            break;
        }
        
        // Comment in for Object
        /*
        density = getAlbedo(pos);
        if(density.a > 0.9) {
            return col;
        }
        col += density;
        t += max(t * 0.001 + density.a * 0.1, 0.015);*/
    }
    
    return col;

}

vec2 boundedMap(vec3 p)
{
    return map(p);
}

vec3 calcNormals(vec3 p)
{
    float epsilon = 0.001;
    return normalize(vec3(boundedMap(p + vec3(epsilon, 0.0, 0.0)).x - boundedMap(p - vec3(epsilon, 0.0, 0.0)).x, boundedMap(p + vec3(0.0, epsilon, 0.0)).x - boundedMap(p - vec3(0.0, epsilon, 0.0)).x, boundedMap(p + vec3(0.0, 0.0, epsilon)).x - boundedMap(p - vec3(0.0, 0.0, epsilon)).x));
    
}

vec4 raycast(vec3 origin, vec3 dir, int maxSteps)
{
    float t = 0.0;
    vec3 p = vec3(0.0);
    vec2 dist = vec2(0.0);
    bool inside = false;
    for(int i = 0; i < maxSteps; ++i)
    {
         p = origin + t * dir;
         dist = boundedMap(p);
//
//        if (dist.x < 0.001) {
//            if(dist.x < -0.001) {
//                return vec4(p + dist.x * dir, dist.y);
//            } else {
//                return vec4(p, dist.y);
//
//            }
//        }
        
        if (abs(dist.x) < 0.001) {
            return vec4(p, dist.y);
        }

        if (dist.x < 0.0) {
            inside = true;
            //return vec4(p + dir * dist.x * 2.0, dist.y);
//
        }
        
        t += dist.x;
        if(t > 40.0) {
            return vec4(0.0, 0.0, 0.0, -200.0);

        }
    }
    
    if(inside) {
        return vec4(p, dist.y);
    }
    
    return vec4(0.0, 0.0, 0.0, -100.0);
}

float softShadow(vec3 origin, vec3 dir, float minT, float maxT, float k)
{
    float res = 1.0;
    float ph = 1e20;

    for(float t = minT; t < maxT; )
    {
        vec3 p = origin + t * dir;
        vec2 dist = map(p);
        if (abs(dist.x) < 0.0001) {
            return 0.0;
        }
        /*
        float y = dist.x*dist.x/(2.0*ph);
        float d = sqrt(dist.x*dist.x-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = dist.x;*/
        res = min( res, k * dist.x / t );
        t += dist.x;

    }
    
    return res;

}
// From https://gamedev.stackexchange.com/questions/59797/glsl-shader-change-hue-saturation-brightness
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float modTime = mod(u_Time, 100.0 * pi);

    float fov = 22.5f;
    float len = distance(u_Ref, u_Eye);
    vec3 look = normalize(u_Ref - u_Eye);
    vec3 right = normalize(cross(look, u_Up));
    float aspect = u_Dimensions.x / u_Dimensions.y;
    vec3 v = u_Up * len * tan(fov);
    vec3 h = right * len * aspect * tan(fov);

    vec3 p = u_Ref + fs_Pos.x * h + fs_Pos.y * v;
    vec3 dir = normalize(p - u_Eye);
    
    vec3 lightPos = vec3(1.0, 0.6, -20.0);
    
    PointLight[3] pointLights;
    pointLights[0] = PointLight(vec3(1.0, 100.0, -20.0), 0.9 * vec3(0.9,0.99,0.98), true, 0.6);
    pointLights[1] = PointLight(vec3(-80.0, 0.6, 20.0), 0.3 * vec3(0.9,0.7,0.7), false, 0.1);
    pointLights[2] = PointLight(vec3(80.0, -40, 20.0), 0.3 * vec3(0.7,0.7,0.9), false, 0.1);

    vec4 isect = raycast(u_Eye, dir, 200);
    float a = getBias((fs_Pos.y + 1.0) * 0.5, 0.68);
    
    // Clear color
    vec3 clearColor = vec3(0.0);
     clearColor = vec3(1.0);

    vec3 albedo = mix(vec3(0.58,0.6,0.72), vec3(0.92,0.95,0.96), a);
    albedo = vec3(0.0);
    vec3 col = albedo;
    float alpha = 1.0;
    if(isect.w >= 0.0)
    {
        col = vec3(0.0);
        vec3 normal = calcNormals(isect.xyz);
        vec3 viewVec = normalize(isect.xyz - u_Eye.xyz);
        float kd = 1.5;
        float ks = 0.9;
        float cosPow = 128.0;
        vec4 albedoAlpha = accumulateDensity(isect.xyz);

        albedo = albedoAlpha.xyz * albedoAlpha.a;
        //albedo = vec3(30.0) * albedoAlpha.a;
        for (int i = 0; i < 3; ++i) {
            vec3 lightVec = normalize(pointLights[i].position - isect.xyz);
            vec3 h = normalize(lightVec - viewVec);
            float diffuse = clamp(dot(normal, lightVec), 0.0, 1.0);
            float specularIntensity = max(pow(max(dot(h, normal), 0.f), cosPow), 0.f);
            specularIntensity = clamp(specularIntensity, 0.0, 1.0);
            
            float shadow = 1.0;
            if (pointLights[i].castsShadow) {
               // shadow = softShadow(isect.xyz + normal * 0.04, lightVec, 0.02, 4.5, 32.0);
            }
            
            vec3 lightIntensity = pointLights[i].ambient + shadow * pointLights[i].color * clamp(kd * diffuse + ks * specularIntensity, 0.0, 2.7);
            col += lightIntensity * albedo;
        }
       // col = albedo;
       // col = mix(col, clearColor, 0.2 * distance(u_Eye, isect.xyz));
        //out_Col = vec4(col, 1.0);
    }
   // vec3 hsv = rgb2hsv(col);
    //hsv.y += 0.10;
    //hsv.z -= 0.1;

   // col = hsv2rgb(hsv);
    out_Col = vec4(col, 1.0);


}
