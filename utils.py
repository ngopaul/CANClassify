import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
import cantools
sns.set()

SIGNAL_EMPTY = 0
SIGNAL_ENUM = 1
SIGNAL_CONTINUOUS = 2
SIGNAL_CLIPPED = 3

SEARCH_UNKNOWN = 100
SEARCH_EMPTY = 101
SEARCH_ENUM = 102
SEARCH_CONTCLIP = 103


def convert_to_big_endian_start(position):
    """
    Convert an integer between 0 and 63 (inclusive) to the equivalent start position if the signal is big endian
    """
    return position // 8 * 8 + (7 - position % 8)


def hex_to_binary(hex_str, pad_to=1, return_np_bool_array=False):
    """ 
    Convert a hex string to a binary string, padded to the given length. 
    
    Args:
        hex_str: a string which represents a hexadecimal number (can have 0x prefix, which is ignored)
        pad_to: how long the returned binary number should be, minimum. If the hex number converted 
            binary is longer than pad_to, then pad_to is ignored. Otherwise, 0's will be added to the
            beginning of the converted binary string until the final string is of the length pad_to.
    Returns:
        str: a binary string equivalent to hex_str's value in hex
    Raises:
        ValueError: invalid literal for int() with base 16, if the given hex_str is invalid
    >>> hex_to_binary("0000430000000091")
    '10000110000000000000000000000000000000010010001'
    >>> hex_to_binary("0000430000000091", 64)
    '0000000000000000010000110000000000000000000000000000000010010001'
    >>> hex_to_binary("0")
    '0'
    >>> hex_to_binary("0", 64)
    '0000000000000000000000000000000000000000000000000000000000000000'
    """
    n = int(hex_str, 16)  
    bStr = ''
    while n > 0: 
        bStr = str(n % 2) + bStr 
        n = n >> 1
    return_string = bStr.zfill(pad_to)
    if not return_np_bool_array:
        return return_string
    return np.array([c == '1' for c in return_string])


def int_to_binary_string(val, width):
    """
    Given an int, return the width-long binary equivalent of the int in the form of a binary string.
    The width should ideally be a multiple of 2, but this is not checked.
    
    Will not fail if the val does not fit into the width. Will just truncate the higher values.
    """
    return str(bin(((1 << width) - 1) & val))[2:].zfill(width)


def mask_hex_str(hex_str, position, length, pad_to=64, byteorder='big', signed=False):
    """
    Returns an decimal which corresponds to the value of the hex string provided, looking only at the the values
    specified by the position and offset.
    If the length is a multiple of 8, then byteorder and signed will be used to obtain the final decimal value.
    :param hex_str: a string which represents a hexadecimal number (can have 0x prefix, which is ignored)
    :param position: a 0-indexed position where the signal starts
    :param length: the length of the signal to look at
    :param pad_to: the size the binary string should be padded with 0s to
    :param byteorder: byteorder of string
    :param signed: whether string is signed or not
    :return: a decimal value which corresponds to the value of the hex_string provided, after converting to binary,
            padding, and masking.
    >>> mask_hex_str("0000430000000091", 0, 64)
    73667279061137
    >>> mask_hex_str("0000430000000091", 0, 64, byteorder='little')
    10448351135503941632
    >>> mask_hex_str("0000430000000091", 0, 64, byteorder='little', signed=True)
    -7998392938205609984
    """
    if byteorder == 'big':
        position = position // 8 * 8 + (7 - position % 8)
    elif byteorder == 'little':
        position = position
        
    if position < 0 or length <= 0:
        raise ValueError("Position and length must be at least 0 and greater than 0 respectively")
    if position + length > pad_to:
        raise ValueError(f"Position {position} and length {length} must be contained in the size of the data, {pad_to}")
    binary_representation = hex_to_binary(hex_str, pad_to=pad_to)[position:position+length]
    round_length_to_8 = (len(binary_representation) + 7) & (-8)
    binary_representation = binary_representation.zfill(round_length_to_8)
    
    length_in_bytes = round_length_to_8 // 8
    byte_representation = int(binary_representation, 2).to_bytes(length_in_bytes, byteorder='big')
    int_representation = int.from_bytes(byte_representation, byteorder=byteorder, signed=signed)
    return int_representation


def mask_bin_str(bin_str, byteorder='big', signed=False):
    binary_representation = bin_str
    round_length_to_8 = (len(binary_representation) + 7) & (-8)
    binary_representation = binary_representation.zfill(round_length_to_8)
    
    length_in_bytes = round_length_to_8 // 8
    byte_representation = int(binary_representation, 2).to_bytes(length_in_bytes, byteorder='big')
    int_representation = int.from_bytes(byte_representation, byteorder=byteorder, signed=signed)
    
    return int_representation


def get_raw_signal_values(csv_data, message_id, position, length, byteorder, bus_limit=10, return_np_bool_array=False,
                          pad_to_at_least=0):
    """
    Return the raw signal values for some csv data, given a message ID, and arguments on how to decode a signal on the
    message. Will return a binary string, or a np boolean array if specified.
    :param csv_data: raw CSV data of CAN signals, requiring columns: MessageID, Message, Time, Bus, MessageLength
    :param message_id: CAN message ID
    :param position: a 0-indexed position where the signal starts
    :param length: the length of the signal to look at
    :param byteorder: byteorder of signal
    :param bus_limit: how many buses to return info from, max
    :param return_np_bool_array: whether or not to return np boolean arrays instead of binary strings
    :param pad_to_at_least: number of bits to pad to
    :return:
    """
    message_data = csv_data.where(csv_data['MessageID'] == message_id)
    buses = message_data['Bus'].unique()
    xs, ys = [], []
    bus_counter = 0
    
    assert 0 <= position <= 63, "Position is 0-indexed location within the message"
    assert byteorder in ['big', 'little'], "byteorder must be either 'big' or 'little'"
    
    if byteorder == 'big':
        position = position // 8 * 8 + (7 - position % 8)
    elif byteorder == 'little':
        position = position
    
    for bus in buses:
        if math.isnan(bus):
            continue
        if bus_counter >= bus_limit:
            break
        message_data_for_bus = message_data.loc[message_data['Bus'] == bus]
        # the pad_to length from "MessageLength" in the csv data is measured in bytes, convert to bits
        pad_to = int(message_data_for_bus["MessageLength"].reset_index(drop=True)[0] * 8)
        pad_to = max(pad_to_at_least, pad_to)
        x = message_data_for_bus['Time']
        y = message_data_for_bus['Message'].apply(
            lambda hex_str: hex_to_binary(hex_str, pad_to=pad_to, return_np_bool_array=return_np_bool_array)[position:position+length])
        xs.append(x)
        ys.append(y)
        bus_counter += 1
    return xs, ys


def plot_message_id(csv_data, message_id, position, length, byteorder='big', signed=False, scale=1, offset=0, plot_fxn=plt.scatter, legend=True):
    message_data = csv_data.where(csv_data['MessageID'] == message_id)
    buses = message_data['Bus'].unique()
    xs, ys = [], []
    
    assert 0 <= position <= 63, "Position is 0-indexed location within the message"
    assert byteorder in ['big', 'little'], "byteorder must be either 'big' or 'little'"
    
    for bus in buses:
        if math.isnan(bus):
            continue
        message_data_for_bus = message_data.loc[message_data['Bus'] == bus]
        # the pad_to length from "MessageLength" in the csv data is measured in bytes, convert to bits
        pad_to = int(message_data_for_bus["MessageLength"].reset_index(drop=True)[0] * 8)
        
        x = message_data_for_bus['Time']
        y = message_data_for_bus['Message'].apply(
                            lambda hex_str: mask_hex_str(hex_str, position, length, pad_to, byteorder, signed)*scale+offset)
        
        if plot_fxn is None:
            pass
        elif plot_fxn == plt.scatter:
            plot_fxn(x, y,
                     label=f"Message {message_id}, bus {bus}, position {position}, "
                           f"length {length}, scale {scale}, offset {offset}",
                     s=1
                     )
        else:
            plot_fxn(x, y,
                     label=f"Message {message_id}, bus {bus}, position {position}, "
                           f"length {length}, scale {scale}, offset {offset}"
                    )
            
        xs.append(x)
        ys.append(y)
        
    if plot_fxn is not None:
        if legend:
            plt.legend(loc='upper left')
        plt.show()
    return xs, ys


def describe_known_signal(cantools_db, message_name, signal_name, csv_data=None, plot_fxn=plt.scatter, silent=False):
    """
    Returns the information known about a given signal. Also plots the signal if given csv_data.
    :param cantools_db:
    :param message_name:
    :param signal_name:
    :param csv_data:
    :param plot_fxn:
    :param silent:
    :return:
    """
    message = cantools_db.get_message_by_name(message_name)
    for signal in message.signals:
        if signal.name == signal_name:
            break
    assert signal.name == signal_name, "This message/signal combination is not in the cantools db."
    if not silent:
        print(f"message name: {message.name}\n"
              f"message frame_id: {message.frame_id}\n"
              f"message is_extended_frame: {message.is_extended_frame}\n"
              f"message length: {message.length}\n"
              f"message comment: {message.comment}\n"
              f"message bus_name: {message.bus_name}\n"
             )
        print(f"signal name: {signal.name}\n"
              f"signal start (1 indexed): {signal.start}\n"
              f"signal length in bits: {signal.length}\n"
              f"signal byte_order: {signal.byte_order}\n"
              f"signal is_signed: {signal.is_signed}\n"
              f"signal is_float: {signal.is_float}\n"
              f"signal scale: {signal.scale}\n"
              f"signal offset: {signal.offset}\n"
              f"signal minimum: {signal.minimum}\n"
              f"signal maximum: {signal.maximum}\n"
              f"signal comment: {signal.comment}\n"
             )
    byte_order = signal.byte_order.replace("_endian", "")
    
    # start = signal.start // 8 * 8 + (7 - signal.start % 8)
    
    # if not silent:
    #     print(f'Actual start: {start}')
    
    if csv_data is not None:
        xs, ys = plot_message_id(csv_data, message.frame_id, signal.start, signal.length, byte_order, signal.is_signed, signal.scale, signal.offset, plot_fxn)
    else:
        xs, ys = None, None
        
    value_dict = {
        "message_length": message.length,
        "frame_id": message.frame_id,
        "start": signal.start,
        "signal_length": signal.length,
        "byte_order": byte_order,
        "is_signed": signal.is_signed,
        "scale": signal.scale,
        "offset": signal.offset,
    }
    
    return xs, ys, value_dict


def calculate_function_metrics(csv_data, message_id, bus, position, length, byteorder='big', signed=False, verbose=False):
    """
    Given csv_data of CAN responses, calculate the function metrics for the given:
    - message_id
    - bus
    - position
    - length
    - byteorder
    - signed
    - scale
    - offset
    """
    message_data = csv_data.where(csv_data['MessageID'] == message_id)
    buses = message_data['Bus'].unique()
    
    if not bus in buses:
        raise ValueError("No bus {bus} in csv_data.")
    
    message_data_for_bus = message_data.loc[message_data['Bus'] == bus]
    # the pad_to length from "MessageLength" in the csv data is measured in bytes, convert to bits
    pad_to = int(message_data_for_bus["MessageLength"].reset_index(drop=True)[0] * 8)

    x = message_data_for_bus['Time']
    y = message_data_for_bus['Message'].apply(
        lambda hex_str: mask_hex_str(hex_str, position, length, pad_to, byteorder, signed))
    y_diff_1 = y.diff()
    y_diff_1_mag = abs(y_diff_1)
    (unique_diff_mag, counts_diff_mag) = np.unique(y_diff_1_mag, return_counts=True)
    most_common_diff_mag = unique_diff_mag[np.argmax(counts_diff_mag)] if len(counts_diff_mag) else 0
    
    y_max = np.max(y)
    y_min = np.min(y)
    epsilon = 2**(length-3)
    delta = y_max - y_min
    y_diff_1_filtered = y_diff_1_mag.where(y_diff_1_mag < delta - epsilon).dropna()
    (unique_diff_mag_filtered, counts_diff_mag_filtered) = np.unique(y_diff_1_filtered, return_counts=True)
    most_common_diff_mag_filtered = unique_diff_mag_filtered[np.argmax(counts_diff_mag_filtered)] if \
                                    len(counts_diff_mag_filtered) else 0
    
    """ VAL1 Continuity """
    # consider using np.max(y_diff_1_mag) vs. most_common_diff_mag
    val_1 = (delta - np.max(y_diff_1_mag))/delta if delta > 0 else 0
    
    """ VAL2 Clipped """
    # consider using np.max(y_diff_1_filtered) vs. most_common_diff_mag_filtered
    # consider using 2**length vs. delta. Currently using 2**length because clippiness should clip to the whole bit range.
    val_2 = (2**length - np.max(y_diff_1_filtered)) / 2**length
    val_2 = val_2 - val_1
    
    """ VAL3 """
    if any(y_diff_1_filtered):
        sigma = np.std(y_diff_1_filtered)
        val_3 = 1 - sigma/(1 + sigma)
    else:
        val_3 = 0
    
    """ VAL4 """
    avg_diff_mags = np.sum(y_diff_1_mag)/len(y_diff_1)
    val_4 = 1 - avg_diff_mags/(1 + avg_diff_mags)
    
    """ VAL5 """
    val_5 = int(np.any(y_diff_1))
    
    if verbose:
        print(f"{message_id} {bus} {position} {length} {byteorder} {'signed' if signed else 'unsigned'} "
              f"Continuity: {val_1} Clipped: {val_2} Counter: {val_3} Enum: {val_4} Changes: {val_5}"
             )
    return val_1, val_2, val_3, val_4, val_5


def get_message_id_length(csv_data, message_id, bus=[0, 1, 2]):
    """Based on a csv of CAN data, return the MessageLength of a given message id on a given bus. """
    first_data = csv_data[(csv_data['MessageID'] == message_id) & (csv_data['Bus'].isin(bus))].iloc[0]
    return int(first_data['MessageLength'])


def select_arbitrary_bit_greedy(unknown_bits):
    """ Given a list of unknown bits (0's represent known, 1 represent unknown), select the first unknown bit (the first 1).
    The index of this bit is i. 
    Return [i, i], (expand_left, expand_right) where expand is a boolean which representat that there is a 1 in that 
    given direction (to the left or right of index i).
    Return [-1, -1], [-1, -1] if there are no 1s left in the array.
    """
    for i in range(len(unknown_bits)):
        if unknown_bits[i] == 1:
            expand_left = 1 if i > 0 and unknown_bits[i-1] == 1 else 0
            expand_right = 1 if i < len(unknown_bits) - 1 and unknown_bits[i+1] == 1 else 0
            return [i, i], [expand_left, expand_right]
    return [-1, -1], [-1, -1]


def greedy_search_message_id(csv_data, message_id, bus, byteorder='big', signed=False, verbose=0):
    """ Given a csv of CAN data, message id, bus, byteorder, and whether to looked for signed or unsigned signals,
    return the predecitions of where signals are. 
    
    Return known_groups.
    
    The structure of known_groups is:
    K -> V
    K: tuple of two numbers: left_index, right_index. These indices are INCLUSIVE and represent where the boundaries of the signal are.
    V: signal_type [SIGNAL_EMPTY, SIGNAL_ENUM, SIGNAL_CONTINUOUS, SIGNAL_CLIPPED], metrics. 
        The metrics are from calculate_function_metrics() being applied to that section of the bits.
        
    known_groups should satisfy the property of labeling EVERYTHING in the signal (every bit in the signal should be labeled).
    Defaults for signals that the algorithm is unsure about are: SIGNAL_EMPTY (if the signal is all 0s there) or SIGNAL_ENUM (if there
        are any 1s in that part of the signal).
    The algorithm will only label a section as SIGNAL_CONTINUOUS or SIGNAL_CLIPPED if it is 'pretty sure' about the signal being of these
        types.
        
    Verbose = 0: no print messages
    Verbose = 1: print only the statuses of the search (i.e. I FOUND this signal)
    Verbose = 2: print both the statuses of the search and the details of calculate_function_metrics() by 
        calling calculate_function_metrics() with verbose=True
    """
    if verbose:
        print(f"Begin greedy search, msg_id: {message_id} bus: {bus} byteorder: {byteorder} signed: {signed}")
    total_message_length = get_message_id_length(csv_data, message_id, bus) * 8
    known_groups = {} # a mapping from left right indices, tuples, to (signal_type, metric). signal_types ie: 
                    # SIGNAL_EMPTY, SIGNAL_ENUM, etc.
    unknown_bits = [1 for _ in range(total_message_length)]
    
    while any(unknown_bits):
        search_indices, expansion_directions = select_arbitrary_bit_greedy(unknown_bits)
        # check if that bit changes
        metrics = calculate_function_metrics(csv_data, message_id, bus, search_indices[0], 1, 
                                             byteorder=byteorder, signed=signed)
        search_mode = SEARCH_UNKNOWN
        previous_contclip_score = 0
        
        # if you cannot expand anymore, then quit with the enum value/0 value
        if expansion_directions == [0, 0]:
            if metrics[4] == 1: # changes
                unknown_bits[search_indices[0]] = 0
                known_groups[tuple(search_indices)] = SIGNAL_ENUM
            else:
                unknown_bits[search_indices[0]] = 0
                known_groups[tuple(search_indices)] = SIGNAL_EMPTY
            break
        
        # have the ability to expand after starting a new search within the message
        while any(expansion_directions):
            if metrics[4] == 1: # the last bits added change value
                if byteorder == 'big':
                    new_bits_to_add = [search_indices[1] + 1]*2 if expansion_directions[1] else [search_indices[0] - 1]*2
                    chosen_expansion_direction = 1 if expansion_directions[1] else 0
                elif byteorder == 'small':
                    # verify that you can expand up to a multiple of 4 bits. Expand to the right first then try the left.
                    current_search_length = search_indices[1] - search_indices[0] + 1
                    expansion_length = 4 - (current_search_length % 4)
                    next_right_index = search_indices[1] + 1
                    right_expansion_test = unknown_bits[next_right_index:next_right_index+expansion_length]
                    can_expand_right = expansion_directions[1] and \
                        len(right_expansion_test) == expansion_length and all(right_expansion_test)
                    next_left_index = search_indices[0] - 1
                    left_expansion_test = unknown_bits[next_left_index-expansion_length:next_left_index]
                    can_expand_left = expansion_directions[0] and \
                        len(left_expansion_test) == expansion_length and all(left_expansion_test)
                    if can_expand_right:
                        new_bits_to_add = [next_right_index,next_right_index+expansion_length]
                        chosen_expansion_direction = 1
                    elif can_expand_left:
                        new_bits_to_add = [next_left_index-expansion_length,next_left_index]
                        chosen_expansion_direction = 0
                    else:
                        expansion_directions = [0, 0]
                        break
                if verbose == 2:
                    print("Calculating added metrics.")
                added_metrics = calculate_function_metrics(csv_data, message_id, bus, new_bits_to_add[0], 
                                                           new_bits_to_add[1] - new_bits_to_add[0] + 1, 
                                                           byteorder=byteorder, signed=signed, verbose=(verbose==2))
                
                if added_metrics[4] == 0: # added bits don't change
                    # throw those bits out, don't expand in that direction
                    expansion_directions[chosen_expansion_direction] = 0
                else: # the added bits do change
                    # get the new search_indices and expansion_directions
                    if chosen_expansion_direction:
                        new_search_indices = [search_indices[0], new_bits_to_add[1]]
                        new_expansion_directions = [expansion_directions[0], 
                                                    int((new_bits_to_add[1] < total_message_length - 1) and \
                                                        unknown_bits[new_bits_to_add[1] + 1])
                                                   ]
                    else:
                        new_search_indices = [new_bits_to_add[0], search_indices[1]]
                        new_expansion_directions = [int((new_bits_to_add[0] > 0) and \
                                                    unknown_bits[new_bits_to_add[0] - 1]),
                                                    expansion_directions[1]
                                                   ]
                    if verbose == 2:
                        print("Calculating new metrics (old + added).")
                    new_metrics = calculate_function_metrics(csv_data, message_id, bus, new_search_indices[0], 
                                                           new_search_indices[1] - new_search_indices[0] + 1, 
                                                           byteorder=byteorder, signed=signed, verbose=(verbose==2))
                    if search_mode == SEARCH_UNKNOWN:
                        # continue the search by passing on new_* to the original *.
                        metrics = new_metrics
                        search_indices = new_search_indices
                        expansion_directions = new_expansion_directions
                        
                        # if you're only looking at two bits so far, don't assume any search_mode yet.
                        if new_search_indices[1] - new_search_indices[0] + 1 <= 2:
                            pass
                        else:
                            # set the search mode to continuous/clipped if the value is greater than 0.5
                            if metrics[0] > 0.5 or metrics[1] > 0.5:
                                if verbose == 2:
                                    print("Setting search mode to SEARCH_CONTCLIP.")
                                search_mode = SEARCH_CONTCLIP
                                previous_contclip_score = max(metrics[0], metrics[1])
                            # or to enum otherwise
                            else:
                                if verbose == 2:
                                    print("Setting search mode to SEARCH_ENUM.")
                                search_mode = SEARCH_ENUM
                    else:
                        if search_mode == SEARCH_CONTCLIP:
                            if max(new_metrics[0], new_metrics[1]) >= previous_contclip_score:
                                metrics = new_metrics
                                search_indices = new_search_indices
                                expansion_directions = new_expansion_directions
                            else:
                                # that expansion was bad, don't expand that direction again
                                expansion_directions[chosen_expansion_direction] = 0
                                # also, don't push the new_* into the the original *, since we don't want that direction
                        elif search_mode == SEARCH_ENUM:
                            # since the bits changed and we're looking for an enum, just push the new_* into the original
                            metrics = new_metrics
                            search_indices = new_search_indices
                            expansion_directions = new_expansion_directions
                        elif search_mode == SEARCH_EMPTY:
                            assert False # you should not be here if you're looking for an empty signal.
            else: # the last bits don't change value.
                if verbose == 2:
                    print("Setting search mode to SEARCH_EMPTY.")
                search_mode = SEARCH_EMPTY
                while expansion_directions[0]:
                    left_metrics = calculate_function_metrics(csv_data, message_id, bus, search_indices[0] - 1, 
                                                           1, byteorder=byteorder, signed=signed)
                    if left_metrics[4] == 0: # nothing changed to the left
                        search_indices[0] -= 1
                        expansion_directions[0] = 1 if search_indices[0] > 0 and unknown_bits[search_indices[0] - 1] else 0
                    else:
                        expansion_directions[0] = 0
                while expansion_directions[1]:
                    right_metrics = calculate_function_metrics(csv_data, message_id, bus, search_indices[1] + 1, 
                                                           1, byteorder=byteorder, signed=signed)
                    if right_metrics[4] == 0: # nothing changed to the right
                        search_indices[1] += 1
                        expansion_directions[1] = 1 if search_indices[1] < total_message_length - 1 and \
                            unknown_bits[search_indices[1] + 1] else 0
                    else:
                        expansion_directions[1] = 0
                        
        # after expanding to the left and right
        signal_type = -1
        if search_mode == SEARCH_UNKNOWN:
            signal_type = SIGNAL_ENUM if metrics[4] else SIGNAL_EMPTY
        elif search_mode == SEARCH_EMPTY:
            signal_type = SIGNAL_EMPTY
        elif search_mode == SEARCH_ENUM:
            signal_type = SIGNAL_ENUM
        elif search_mode == SEARCH_CONTCLIP:
            signal_type = SIGNAL_CONTINUOUS if metrics[0] > metrics[1] else SIGNAL_CLIPPED
        newfound_indices = tuple(search_indices)
        newfound_value = (signal_type, 
                           calculate_function_metrics(csv_data, message_id, bus, 
                                                      search_indices[0], 
                                                      search_indices[1] - search_indices[0] + 1,
                                                      byteorder=byteorder, signed=signed)
                         )
        if verbose > 0:
            print(f'Found new indices: {newfound_indices}, value: {newfound_value}')
        known_groups[newfound_indices] = newfound_value
        for i in range(search_indices[0], search_indices[1] + 1):
            unknown_bits[i] = 0
    return known_groups