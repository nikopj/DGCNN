using EzXML
using ImageView
using ImageFiltering
using Statistics
using DelimitedFiles

cloudFile = "/home/brian/Documents/speakerData/speaker/pointCloud.csv"

doc = readxml("/home/brian/Documents/speakerData/speaker/Header.xml")
ocity = root(doc)

global pixelDims = [0,0,0]
global pixelWidth = [0.,0.,0.]
global axOrder = ""

for node in eachelement(ocity)
	if node.name == "Image"
		for node2 in eachelement(node)
		    if node2.name == "SizePixel"
                i = 1
                for c in split(node2.content)
                    global pixelDims[i] = parse(Int64,c)
                    i = i+1
                end
            
            elseif node2.name == "AxisOrder"
                global axOrder = node2.content

            elseif node2.name == "PixelSpacing"
                i = 1
                for c in split(node2.content)
                    global pixelWidth[i] = parse(Float32,c)*1000.
                    i = i+1
                end

            end
		end
	end
end

numEls = prod(pixelDims)

Intensity = Array{Float32}(undef, numEls) # you have to preallocate to read in correctly

filename = "/home/brian/Documents/speakerData/speaker/data/Intensity.data"
f = open(filename)

read!(f, Intensity)

close(f)

T = 50

Intensity = reshape(Intensity,tuple(pixelDims...))
Intensity[Intensity.<T] .= 0
Intensity[Intensity.!=0] .= 1
numPts = Int64(sum(Intensity))

global pointCloud =  Array{Float32}(undef, numPts,3) # going to be a happy little cloud
global N = 1
for i = 1:pixelDims[1]
    for j = 1:pixelDims[2]
        for k = 1:pixelDims[3]
            if Intensity[i,j,k] != 0
                global point = [i,j,k].*pixelWidth
                pointCloud[N,:] = point
                global N = N+1
            end
        end
    end
end

writedlm(cloudFile,pointCloud)
