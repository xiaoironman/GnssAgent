# This file contains the wrapper functions for the RTKLIB tools used in the project.
import json
import subprocess
import os

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from pydantic.v1 import BaseModel, Field
from typing import List, Optional
from langchain_core.tools import tool

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt


# GPT_MODEL = "gpt-3.5-turbo-0125"
GPT_MODEL = "gpt-4o"
openai_key_rtkflow = os.environ.get("OPENAI_KEY_RTKFLOW")
CACHE_FOLDER = r"C:\Users\Xiao Liu\Downloads\GnssAgent\cache"
RTKLIB_FOLDER = r"C:\GNSS\RTKLIB_bin-rtklib_2.4.3\bin"

SYSTEM_PROMPT = "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous. Maximum 1 tool call per user message. If there is any output file being generated, put it in the following cache folder: " + CACHE_FOLDER


class Rnx2RtkpParams(BaseModel):
    input_files: List[str] = Field(..., description="List of files. The first file should be the rover RINEX OBS file. Then the base station file. And finally the navigation files and other RINEX NAV/GNAV/HNAVsp3/CLK files, etc.")
    extra_commands: Optional[str] = Field(None,
                                          description="""Extra commands to be passed to the rnx2rtkp executable.
 Command options are as follows ([]:default). With -k option, the processing options are input from the 
 configuration file. In this case, command line options precede options in the configuration file.
 -k file   input options from configuration file [off]
 -o file   set output file [stdout]
 -ts ds ts start day/time (ds=y/m/d ts=h:m:s) [obs start time]
 -te de te end day/time   (de=y/m/d te=h:m:s) [obs end time]
 -ti tint  time interval (sec) [all]
 -p mode   mode (0:single,1:dgps,2:kinematic,3:static,4:moving-base, 5:fixed,6:ppp-kinematic,7:ppp-static) [2]
 -m mask   elevation mask angle (deg) [15]
 -sys s[,s...] nav system(constellations) (s=G:GPS,R:GLO,E:GAL,J:QZS,C:BDS,I:IRN) [G|R]
 -f freq   number of frequencies for relative mode (1:L1,2:L1+L2,3:L1+L2+L5) [2]
 -v thres  validation threshold for integer ambiguity (0.0:no AR) [3.0]
 -b        backward solutions [off]
 -c        forward/backward combined solutions [off]
 -i        instantaneous integer ambiguity resolution [off]
 -h        fix and hold for integer ambiguity resolution [off]
 -e        output x/y/z-ecef position [latitude/longitude/height]
 -a        output e/n/u-baseline [latitude/longitude/height]
 -n        output NMEA-0183 GGA sentence [off]
 -g        output latitude/longitude in the form of ddd mm ss.ss' [ddd.ddd]
 -t        output time in the form of yyyy/mm/dd hh:mm:ss.ss [sssss.ss]
 -u        output time in utc [gpst]
 -d col    number of decimals in time [3]
 -s sep    field separator [' ']
 -r x y z  reference (base) receiver ecef pos (m) [average of single pos]
           rover receiver ecef pos (m) for fixed or ppp-fixed mode
 -l lat lon hgt reference (base) receiver latitude/longitude/height (deg/m)
           rover latitude/longitude/height for fixed or ppp-fixed mode
 -y level  output solution status (0:off,1:states,2:residuals) [0]
 -x level  debug trace level (0:off) [0]
 """)


@tool(args_schema=Rnx2RtkpParams)
def rnx2rtkp(input_files: List[str], extra_commands: str = None):
    """
 Read RINEX OBS/NAV/GNAV/HNAV/CLK, SP3, SBAS message log files and ccompute
 receiver (rover) positions and output position solutions.
 The first RINEX OBS file shall contain receiver (rover) observations. For the
 relative mode, the second RINEX OBS file shall contain reference
 (base station) receiver observations. At least one RINEX NAV/GNAV/HNAV
 file shall be included in input files. To use SP3 precise ephemeris, specify
 the path in the files. The extension of the SP3 file shall be .sp3 or .eph.
 All of the input file paths can include wild-cards (*). To avoid command
 line deployment of wild-cards, use "..." for paths with wild-cards.
    """
    rnx2rtkp_path = os.path.join(RTKLIB_FOLDER, "rnx2rtkp.exe")
    command = [rnx2rtkp_path]
    command += input_files
    if extra_commands:
        command += extra_commands.split()

    if extra_commands and '-o' in extra_commands:
        output_file = extra_commands.split("-o")[1].strip()
    else:
        output_file = os.path.join(os.path.dirname(input_files[0]),
                                   os.path.basename(input_files[0]).split(".")[0] + "_pvt_output.pos")

    command += ["-o", output_file]
    result = subprocess.run(command, capture_output=True)

    output_str = "The output from rnx2rtkp executable is as follows:\n"
    if b'processing' in result.stderr:
        output_str += f"The generated PVT output file is successfully generated at: {output_file}\n"
    else:
        output_str += f"Standard error:\n{result.stderr}\n"
    return output_str


class ConvBinParams(BaseModel):
    file_to_convert: str = Field(..., description='The target file to be converted into RINEX format')
    options: Optional[str] = Field(None,
                                   description="""Extra commands to be passed to the convbin executable [default].
                                     -ts y/m/d h:m:s  start time [all]
                                     -te y/m/d h:m:s  end time [all]
                                     -tr y/m/d h:m:s  approximated time for RTCM/CMR/CMR+ messages
                                     -ti tint     observation data interval (s) [all]
                                     -tt ttol     observation data epoch tolerance (s) [0.005]
                                     -span span   time span (h) [all]
                                     -r format    log format type
                                                  rtcm2= RTCM 2
                                                  rtcm3= RTCM 3
                                                  nov  = NovAtel OEMV/4/6,OEMStar
                                                  oem3 = NovAtel OEM3
                                                  ubx  = ublox LEA-4T/5T/6T/7T/M8T
                                                  ss2  = NovAtel Superstar II
                                                  hemis= Hemisphere Eclipse/Crescent
                                                  stq  = SkyTraq S1315F
                                                  javad= Javad
                                                  nvs  = NVS NV08C BINR
                                                  binex= BINEX
                                                  rt17 = Trimble RT17
                                                  sbf  = Septentrio SBF
                                                  cmr  = CMR/CMR+
                                                  tersus= TERSUS
                                                  rinex= RINEX
                                     -ro opt      receiver options
                                     -f freq      number of frequencies [3]
                                     -hc comment  rinex header: comment line
                                     -hm marker   rinex header: marker name
                                     -hn markno   rinex header: marker number
                                     -ht marktype rinex header: marker type
                                     -ho observ   rinex header: oberver name and agency separated by /
                                     -hr rec      rinex header: receiver number, type and version separated by /
                                     -ha ant      rinex header: antenna number and type separated by /
                                     -hp pos      rinex header: approx position x/y/z separated by /
                                     -hd delta    rinex header: antenna delta h/e/n separated by /
                                     -v ver       rinex version [2.11]
                                     -od          include doppler frequency in rinex obs [off]
                                     -os          include snr in rinex obs [off]
                                     -oi          include iono correction in rinex nav header [off]
                                     -ot          include time correction in rinex nav header [off]
                                     -ol          include leap seconds in rinex nav header [off]
                                     -scan        scan input file [on]
                                     -noscan      no scan input file [off]
                                     -halfc       half-cycle ambiguity correction [off]
                                     -mask   [sig[,...]] signal mask(s) (sig={G|R|E|J|S|C|I}L{1C|1P|1W|...})
                                     -nomask [sig[,...]] signal no mask (same as above)
                                     -x sat       exclude satellite
                                     -y sys       exclude systems (G:GPS,R:GLO,E:GAL,J:QZS,S:SBS,C:BDS,I:IRN)
                                     -d dir       output directory [same as input file]
                                     -c staid     use RINEX file name convention with staid [off]
                                     -o ofile     output RINEX OBS file
                                     -n nfile     output RINEX NAV file
                                     -g gfile     output RINEX GNAV file
                                     -h hfile     output RINEX HNAV file
                                     -q qfile     output RINEX QNAV file
                                     -l lfile     output RINEX LNAV file
                                     -b cfile     output RINEX CNAV file
                                     -i ifile     output RINEX INAV file
                                     -s sfile     output SBAS message file
                                     -trace level output trace level [off]

                                     If any output file specified, default output files (<file>.obs,
                                     <file>.nav, <file>.gnav, <file>.hnav, <file>.qnav, <file>.lnav and
                                     <file>.sbs) are used.
                                    
                                     If receiver type is not specified, type is recognized by the input
                                     file extension as follows.
                                         *.rtcm2       RTCM 2
                                         *.rtcm3       RTCM 3
                                         *.gps         NovAtel OEMV/4/6,OEMStar
                                         *.ubx         u-blox LEA-4T/5T/6T/7T/M8T
                                         *.log         NovAtel Superstar II
                                         *.bin         Hemisphere Eclipse/Crescent
                                         *.stq         SkyTraq S1315F
                                         *.jps         Javad
                                         *.bnx,*binex  BINEX
                                         *.rt17        Trimble RT17
                                         *.sbf         Septentrio SBF
                                         *.cmr         CMR/CMR+
                                         *.trs         TERSUS
                                         *.obs,*.*o    RINEX OBS
                                   
                                     If receiver type is not recognized, type is set to RINEX.""")


@tool(args_schema=ConvBinParams)
def convbin(file_to_convert: str, options: str = None):
    """
     Convert RTCM, receiver raw data log and RINEX file to RINEX and SBAS/LEX
     message file. SBAS message file complies with RTKLIB SBAS/LEX message
     format. It supports the message or file types of the major GNSS receiver manufacturers.
    """
    convbin_path = os.path.join(RTKLIB_FOLDER, "convbin.exe")
    command = [convbin_path, file_to_convert]
    if options:
        command += options.split()

    result = subprocess.run(command, capture_output=True)
    output_str = "The output from convbin executable is as follows:\n"
    if result.returncode == 0 and not 'error' in result.stderr.decode():
        output_str += f"The conversion is successful! Output files are in folder {os.path.dirname(file_to_convert)} \n"
    else:
        output_str += f"Standard error:\n{result.stderr}\n"
    return output_str


ALL_TOOLS = [rnx2rtkp, convbin]

client = OpenAI(api_key=openai_key_rtkflow)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


if __name__ == "__main__":
    # create the tool executor object to call the tools
    tool_executor = ToolExecutor(ALL_TOOLS)
    # convert StructuredTool objects to OpenAI tool in dicts for later use
    ALL_TOOLS = [convert_to_openai_tool(tool) for tool in ALL_TOOLS]
    rinex_path = r"C:\Users\Xiao Liu\Downloads\GnssAgent\test data\combined.20o"
    messages = []
    messages.append({"role": "system",
                     "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": "I want to compute the PVT solution of a RINEX file. Can you help me?"})
    chat_response = chat_completion_request(
        messages, tools=ALL_TOOLS
    )

    messages.append(chat_response.choices[0].message)
    messages.append({"role": "user", "content": "The rover rinex file is at: " + rinex_path + ". The navigation files are in the same folder and all starting with the characters PLAN, the base station rinex file is UPC_combined.rnx that is also in the same folder."})

    chat_response = chat_completion_request(
        messages, tools=ALL_TOOLS
    )

    # if response is a function call, then call the tool
    if chat_response.choices[0].finish_reason == "tool_calls":
        tool_call = chat_response.choices[0].message.tool_calls[0]
        function = tool_call.function
        function_name = function.name
        _tool_input = function.arguments
        print(f"function_name: {function_name};\nfunction arguments: {_tool_input}")
    else:
        print(chat_response.choices[0].message.content)

    # call the tool
    action = ToolInvocation(
        tool=function_name,
        tool_input=json.loads(_tool_input),
    )

    response = tool_executor.invoke(action)
    print(response)
