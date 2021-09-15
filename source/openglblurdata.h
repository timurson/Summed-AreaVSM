#ifndef OPENGLBLURDATA_H
#define OPENGLBLURDATA_H OpenGLBlurData

class OpenGLBlurData
{
public:
    OpenGLBlurData(int width, float deviation = 1.0f);
    int blurWidth;
    int blurWidth2;
    float weights[65];
};

#endif // OPENGLBLURDATA_H
