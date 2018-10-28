# 00-data.jl
# Extracts audio features from TIMIT to be used in speech recognition

using Flux: onehotbatch
using WAV
using BSON

# This wookay's fork of MFCC updated to work with Julia v0.7/1.0
# https://github.com/wookay/MFCC.jl
using MFCC

# Define constants that will be used
const TRAINING_DATA_DIR = "TIMIT/TRAIN"
const TEST_DATA_DIR = "TIMIT/TEST"

const TRAINING_OUT_DIR = "train"
const TEST_OUT_DIR = "test"

# Make dictionary to map from phones to class numbers
const PHONES = split("h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng")
translations = Dict(phone=>i for (i, phone) in enumerate(PHONES))
translations["sil"] = translations["h#"]
const PHONE_TRANSLATIONS = translations

# Make dictionary to perform class folding
const FOLDINGS = Dict(
  "ao" => "aa",
  "ax" => "ah",
  "ax-h" => "ah",
  "axr" => "er",
  "hv" => "hh",
  "ix" => "ih",
  "el" => "l",
  "em" => "m",
  "en" => "n",
  "nx" => "n",
  "eng" => "ng",
  "zh" => "sh",
  "pcl" => "sil",
  "tcl" => "sil",
  "kcl" => "sil",
  "bcl" => "sil",
  "dcl" => "sil",
  "gcl" => "sil",
  "h#" => "sil",
  "pau" => "sil",
  "epi" => "sil",
  "ux" => "uw"
)

FRAME_LENGTH = 0.025 # ms
FRAME_INTERVAL = 0.010 # ms

"""
  makeFeatures(wavFname, phnFname)

Extracts Mel filterbanks and associated labels from `wavFname` and `phnFaname`.
"""
function makeFeatures(phnFname, wavFname)
  samps, sr = wavread(wavFname)
  samps = vec(samps)

  mfccs, _, _ = mfcc(samps, sr, :rasta; wintime=FRAME_LENGTH, steptime=FRAME_INTERVAL)

  local lines
  open(phnFname, "r") do f
    lines = readlines(f)
  end

  boundaries = Vector()
  labels = Vector()

  # first field in the file is the beginning sample number, which isn't
  # needed for calculating where the labels are
  for line in lines
    _, boundary, label = split(line)
    boundary = parse(Int64, boundary)
    push!(boundaries, boundary)
    push!(labels, label)
  end

  labelInfo = collect(zip(boundaries, labels))
  labelInfoIdx = 1
  boundary, label = labelInfo[labelInfoIdx]
  nSegments = length(labelInfo)

  frameLengthSamples = FRAME_LENGTH * sr
  frameIntervalSamples = FRAME_INTERVAL * sr
  halfFrameLength = FRAME_LENGTH / 2

  # Begin generating sequence labels by looping through the MFCC
  # frames

  labelSequence = Vector() # Holds the sequence of labels

  idxsToDelete = Vector() # To store indices for frames labeled as 'q'
  for i=1:size(mfccs, 1)
    win_end = frameLengthSamples + (i-1)*frameIntervalSamples

    # Move on to next label if current frame of samples is more than half
    # way into next labeled section and there are still more labels to
    # iterate through
    if labelInfoIdx < nSegments && win_end - boundary > halfFrameLength

      labelInfoIdx += 1
      boundary, label = labelInfo[labelInfoIdx]
    end

    if label == "q"
      push!(idxsToDelete, i)
      continue
    end

    push!(labelSequence, label)
  end

  # Remove the frames that were labeld as 'q'
  mfccs = mfccs[[i for i in 1:size(mfccs,1) if !(i in Set(idxsToDelete))],:]

  mfccDeltas = deltas(mfccs, 2)
  features = hcat(mfccs, mfccDeltas)
  return (features, labelSequence)
end

"""
  createData(data_dir, out_dir)

Extracts data from files in `data_dir` and saves results in `out_dir`.
"""
function createData(data_dir, out_dir)

  ! isdir(out_dir) && mkdir(out_dir)

  for (root, dirs, files) in walkdir(data_dir)

    # Exclude the files that are part of the speaker accent readings
    files = [x for x in files if ! occursin("SA", x)]

    phnFnames = [x for x in files if occursin("PHN", x)]
    wavFnames = [x for x in files if occursin("WAV", x)]

    one_dir_up = basename(root)
    print("$(root)\r")

    for (wavFname, phnFname) in zip(wavFnames, phnFnames)
      phn_path = joinpath(root, phnFname)
      wav_path = joinpath(root, wavFname)

      x, y = makeFeatures(phn_path, wav_path)

      # Generate class nums; there are 61 total classes, but only 39 are
      # used after folding.
      y = [PHONE_TRANSLATIONS[x] for x in y]
      class_nums = [n for n in 1:61]
      y = onehotbatch(y, class_nums)

      base, _ = splitext(phnFname)
      dat_name = one_dir_up * base * ".bson"
      dat_path = joinpath(out_dir, dat_name)
      BSON.@save dat_path x y
    end
  end
  println()
end

createData(TRAINING_DATA_DIR, TRAINING_OUT_DIR)
createData(TEST_DATA_DIR, TEST_OUT_DIR)
