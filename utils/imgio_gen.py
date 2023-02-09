#!/usr/bin/env python3.4

import os
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
import OpenEXR
import sys
import cv2

from builtins import *

import Imath
from collections import defaultdict

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
HALF  = Imath.PixelType(Imath.PixelType.HALF)
UINT  = Imath.PixelType(Imath.PixelType.UINT)

NO_COMPRESSION    = Imath.Compression(Imath.Compression.NO_COMPRESSION)
RLE_COMPRESSION   = Imath.Compression(Imath.Compression.RLE_COMPRESSION)
ZIPS_COMPRESSION  = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
ZIP_COMPRESSION   = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
PIZ_COMPRESSION   = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
PXR24_COMPRESSION = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)

NP_PRECISION = {
  "FLOAT": np.float32,
  "HALF":  np.float16,
  "UINT":  np.uint32
}


def open(filename):
  # Check if the file is an EXR file
  if not OpenEXR.isOpenExrFile(filename):
    raise Exception("File '%s' is not an EXR file." % filename)
  # Return an `InputFile`
  return InputFile(OpenEXR.InputFile(filename), filename)


def read(filename, channels = "default", precision = FLOAT):
  f = open(filename)
  if _is_list(channels):
    # Construct an array of precisions
    return f.get_dict(channels, precision=precision)

  else:
    return f.get(channels, precision)

def read_all(filename, precision = FLOAT):
  f = open(filename)
  return f.get_all(precision=precision)

def write(filename, data, channel_names = None, precision = FLOAT, compression = PIZ_COMPRESSION, extra_headers={}):

  # Helper function add a third dimension to 2-dimensional matrices (single channel)
  def make_ndims_3(matrix):
    if matrix.ndim > 3 or matrix.ndim < 2:
      raise Exception("Invalid number of dimensions for the `matrix` argument.")
    elif matrix.ndim == 2:
      matrix = np.expand_dims(matrix, -1)
    return matrix

  # Helper function to read channel names from default
  def get_channel_names(channel_names, depth):
    if channel_names:
      if depth != len(channel_names):
        raise Exception("The provided channel names have the wrong length (%d vs %d)." % (len(channel_names), depth))
      return channel_names
    elif depth in _default_channel_names:
      return _default_channel_names[depth]
    else:
      raise Exception("There are no suitable default channel names for data of depth %d" % depth)

  #
  # Case 1, the `data` argument is a dictionary
  #
  if isinstance(data, dict):
    # Make sure everything has ndims 3
    for group, matrix in data.items():
      data[group] = make_ndims_3(matrix)

    # Prepare precisions
    if not isinstance(precision, dict):
      precisions = {group: precision for group in data.keys()}
    else:
      precisions = {group: precision.get(group, FLOAT) for group in data.keys()}

    # Prepare channel names
    if channel_names is None:
      channel_names = {}
    channel_names = {group: get_channel_names(channel_names.get(group), matrix.shape[2]) for group, matrix in data.items()}

    # Collect channels
    channels = {}
    channel_data = {}
    width = None
    height = None
    for group, matrix in data.items():
      # Read the depth of the current group
      # and set height and width variables if not set yet
      if width is None:
        height, width, depth = matrix.shape
      else:
        depth = matrix.shape[2]
      names = channel_names[group]
      # Check the number of channel names
      if len(names) != depth:
        raise Exception("Depth does not match the number of channel names for channel '%s'" % group)
      for i, c in enumerate(names):
        if group == "default":
          channel_name = c
        else:
          channel_name = "%s.%s" % (group, c)
        channels[channel_name] = Imath.Channel(precisions[group])
        channel_data[channel_name] = matrix[:,:,i].astype(NP_PRECISION[str(precisions[group])]).tobytes()

    # Save
    header = OpenEXR.Header(width, height)
    if extra_headers:
      header = dict(header, **extra_headers)
    header['compression'] = compression
    header['channels'] = channels
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels(channel_data)

  #
  # Case 2, the `data` argument is one matrix
  #
  elif isinstance(data, np.ndarray):
    data = make_ndims_3(data)
    height, width, depth = data.shape
    channel_names = get_channel_names(channel_names, depth)
    header = OpenEXR.Header(width, height)
    if extra_headers:
      header = dict(header, **extra_headers)
    header['compression'] = compression
    header['channels'] = {c: Imath.Channel(precision) for c in channel_names}
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({c: data[:,:,i].astype(NP_PRECISION[str(precision)]).tobytes() for i, c in enumerate(channel_names)})

  else:
    raise Exception("Invalid precision for the `data` argument. Supported are NumPy arrays and dictionaries.")


def tonemap(matrix, gamma=2.2):
  return np.clip(matrix ** (1.0/gamma), 0, 1)


class InputFile(object):

  def __init__(self, input_file, filename=None):
    self.input_file = input_file

    if not input_file.isComplete():
      raise Exception("EXR file '%s' is not ready." % filename)

    header = input_file.header()
    dw     = header['dataWindow']

    self.width             = dw.max.x - dw.min.x + 1
    self.height            = dw.max.y - dw.min.y + 1
    self.channels          = sorted(header['channels'].keys(),key=_channel_sort_key)
    self.depth             = len(self.channels)
    self.precisions        = [c.type for c in header['channels'].values()]
    self.channel_precision = {c: v.type for c, v in header['channels'].items()}
    self.channel_map       = defaultdict(list)
    self.root_channels     = set()
    self._init_channel_map()

  def _init_channel_map(self):
    # Make a dictionary of subchannels per channel
    for c in self.channels:
      self.channel_map['all'].append(c)
      parts = c.split('.')
      if len(parts) == 1:
        self.root_channels.add('default')
        self.channel_map['default'].append(c)
      else:
        self.root_channels.add(parts[0])
      for i in range(1, len(parts)+1):
        key = ".".join(parts[0:i])
        self.channel_map[key].append(c)

  def describe_channels(self):
    if 'default' in self.root_channels:
      for c in self.channel_map['default']:
        print (c)
    for group in sorted(list(self.root_channels)):
      if group != 'default':
        channels = self.channel_map[group]
        print("%-20s%s" % (group, ",".join([c[len(group)+1:] for c in channels])))

  def get(self, group = 'default', precision=FLOAT):
    channels = self.channel_map[group]

    if len(channels) == 0:
      print("I did't find any channels in group '%s'." % group)
      print("You could try:")
      self.describe_channels()
      raise Exception("I did't find any channels in group '%s'." % group)

    strings = self.input_file.channels(channels)

    matrix = np.zeros((self.height, self.width, len(channels)), dtype=NP_PRECISION[str(precision)])
    for i, string in enumerate(strings):
      precision = NP_PRECISION[str(self.channel_precision[channels[i]])]
      matrix[:,:,i] = np.frombuffer(string, dtype = precision) \
                        .reshape(self.height, self.width)
    return matrix

  def get_all(self, precision = {}):
    return self.get_dict(self.root_channels, precision)

  def get_dict(self, groups = [], precision = {}):

    if not isinstance(precision, dict):
      precision = {group: precision for group in groups}

    return_dict = {}
    todo = []
    for group in groups:
      group_chans = self.channel_map[group]
      if len(group_chans) == 0:
        print("I didn't find any channels for the requested group '%s'." % group)
        print("You could try:")
        self.describe_channels()
        raise Exception("I did't find any channels in group '%s'." % group)
      if group in precision:
        p = precision[group]
      else:
        p = FLOAT
      matrix = np.zeros((self.height, self.width, len(group_chans)), dtype=NP_PRECISION[str(p)])
      return_dict[group] = matrix
      for i, c in enumerate(group_chans):
        todo.append({'group': group, 'id': i, 'channel': c})

    if len(todo) == 0:
      print("Please ask for some channels, I cannot process empty queries.")
      print("You could try:")
      self.describe_channels()
      raise Exception("Please ask for some channels, I cannot process empty queries.")

    strings = self.input_file.channels([c['channel'] for c in todo])

    for i, item in enumerate(todo):
      precision = NP_PRECISION[str(self.channel_precision[todo[i]['channel']])]
      return_dict[item['group']][:,:,item['id']] = \
          np.frombuffer(strings[i], dtype = precision) \
            .reshape(self.height, self.width)
    return return_dict


def _sort_dictionary(key):
  if key == 'R' or key == 'r':
    return "000010"
  elif key == 'G' or key == 'g':
    return "000020"
  elif key == 'B' or key == 'b':
    return "000030"
  elif key == 'A' or key == 'a':
    return "000040"
  elif key == 'X' or key == 'x':
    return "000110"
  elif key == 'Y' or key == 'y':
    return "000120"
  elif key == 'Z' or key == 'z':
    return "000130"
  else:
    return key


def _channel_sort_key(i):
  return [_sort_dictionary(x) for x in i.split(".")]


_default_channel_names = {
  1: ['Z'],
  2: ['X','Y'],
  3: ['R','G','B'],
  4: ['R','G','B','A']
}


def _is_list(x):
  return isinstance(x, (list, tuple, np.ndarray))

def readgen(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    elif file.endswith('.exr'): return open(file).get() #https://github.com/tvogels/pyexr
    else: raise Exception('don\'t know how to read %s' % file)

def writegen(file, data):
    if file.endswith('.float3'): return writeFloat(file, data)
    elif file.endswith('.flo'): return writeFlow(file, data)
    elif file.endswith('.ppm'): return writeImage(file, data)
    elif file.endswith('.pgm'): return writeImage(file, data)
    elif file.endswith('.png'): return writeImage(file, data)
    elif file.endswith('.jpg'): return writeImage(file, data)
    elif file.endswith('.pfm'): return writePFM(file, data)
    elif file.endswith('.exr'): return write(file, data) #https://github.com/tvogels/pyexr
    else: raise Exception('don\'t know how to write %s' % file)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return cv2.imread(name)

def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return misc.imsave(name, data)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data

def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)
