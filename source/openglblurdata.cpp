#include "openglblurdata.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

const float Sqrt2Pi = 2.5066282746310005024157652848110452530069867406099383f;

float normalDist(float value, float mean, float deviation)
{
    value -= mean;
    float valueSquared = value * value;
    float variance = deviation * deviation;
    return glm::exp(-valueSquared / (2.0f * variance)) / (Sqrt2Pi * deviation);
}

OpenGLBlurData::OpenGLBlurData(int width, float deviation) 
    : blurWidth(width)
{
    float total = 0.0f, current;

    // Make sure the BlurData is within the valid range.
    if (blurWidth < 1) blurWidth = 1;
    if (blurWidth > 32) blurWidth = 32;
    blurWidth2 = 2 * blurWidth;

    // Calculate the original normal distribution.
    for (int i = 0; i < blurWidth; ++i)
    {
        current = normalDist(float(blurWidth - i), 0.0f, deviation);
        weights[i] = weights[(blurWidth2 - i)] = current;
        total += 2.0f * current;
    }
    weights[blurWidth] = normalDist(0.0f, 0.0f, deviation);
    total += weights[blurWidth];

    // Normalize the values so that they sum to 1
    for (int i = 0; i <= blurWidth2; ++i)
    {
        weights[i] /= total;
    }
}