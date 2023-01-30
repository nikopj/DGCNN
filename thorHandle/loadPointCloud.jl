using DelimitedFiles
using WGLMakie

cloudFile = "/home/brian/Documents/speakerData/speaker/pointCloud.csv"

pointCloud = readdlm(cloudFile)

pts = pointCloud[1:100:end,:]


scatter(pts[:,1],pts[:,2],pts[:,3])
