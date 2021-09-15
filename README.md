# Summed-Area Soft Variance Shadows
I first learned of Variance Shadow Mapping technique from the [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps) article back when I was an undergrad.  At
the time I thought it was a pretty neat way of generating shadows that produced some nice results.  The part that I didn't quite understand at the time was how to generate summed-area shadow textures and use them
to produce soft shadows.  Now that I have more experience with graphics API under my belt, I decided to revisit this topic and to implement Summed-Area Soft Variance Shadows in all of their natural glory.

## Main Features:
*  Deferred shading with Real-Time soft variance shadow utilizing summed-area table.
*  Summed-area textures can be generated using compute shader or using Hensley's recursive doubling method.
*  Percentage-Closer Soft Shadows are implemented as was described in Fernando's 2005 paper.
*  Run-time debugging of shadow map and light configuration thru the utilization of Dear ImGui library.

## Things I Learned:
*  Don’t go above 1K for the size of your SATs or your shadow will fail!
*  Summed-Area Tables will greatly magnify all the problems that filtered shadow techniques have.
*  Filtering using SATs is fast and very good quality, but works much better for larger kernel sizes.
*  Floating-point precision caused a lot of issue.

![Alt Text](https://github.com/timurson/Summed-AreaVSM/blob/master/Image1.PNG)
![Alt Text](https://github.com/timurson/Summed-AreaVSM/blob/master/Image2.PNG)
![Alt Text](https://github.com/timurson/Summed-AreaVSM/blob/master/Image3.PNG)

# License
Copyright (C) 2021 Roman Timurson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
