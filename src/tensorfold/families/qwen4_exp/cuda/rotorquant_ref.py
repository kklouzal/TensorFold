"""Pinned RotorQuant-family format tables and independent cache-codec references.

This is TensorFold-owned integration of scalar Lloyd-Max quantization and ordinary
orthogonal block rotations, rather than a copy of RotorQuant's cache or kernels.
Format v1 stores centroid indices plus one FP32 scale per128 components. The
descriptor identifies original-RMS or reconstruction-norm scale policy, and
optional outlier scaling of the coordinates used for index selection. Original
norms precede the block rotation. Queries use the same forward rotation; weighted
values use its transpose after attention. The source dimension must be a multiple
of 128.

The coefficients below are actual IEEE754 binary32 values, not RNG seeds. The
Planar table repeats four adjacent-pair rotations. Isofast is left multiplication
by the exactly unit quaternion (1/2,1/2,1/2,1/2). Neither mixes all 128 coordinates
or promises a Haar-rotated Gaussian input distribution. Lloyd-Max tables minimize
scalar error for N(0,1); actual model quality requires separate validation.

The Torch reference deliberately stages FP32 operations. The stdlib FP64 oracle
uses dense matrices and independent bit loops. References are validation tools,
not serving fallbacks; no Torch, SciPy or accelerator is imported at module load.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence

GROUP = 128
VERSION = 1
NORM8_UID = "candidate-classic8-reconstruction-rms-v1-54219992dc1919739eeaab5e43f0d085edfaf1a050b1fe0e6dc104d75fd147a4"
NORM8_SOURCE_BOUND = 2**30
NORM8_METADATA_BOUND = 128 * NORM8_SOURCE_BOUND
PLANAR_COS = (0.9238795042037964, 0.3826834261417389, -0.3826834261417389, -0.9238795042037964)
PLANAR_SIN = (0.3826834261417389, 0.9238795042037964, 0.9238795042037964, 0.3826834261417389)
ISO_QUATERNION = (0.5, 0.5, 0.5, 0.5)

# Analytic normal-cell expectations solved to <2e-14 fixed-point residual, then
# rounded once to binary32. No finite-range quadrature or SciPy is required.
CENTROIDS_3 = (
    -2.1519455909729004, -1.3439092636108398, -0.7560052871704102, -0.2450941801071167,
    0.2450941801071167, 0.7560052871704102, 1.3439092636108398, 2.1519455909729004,
)
CENTROIDS_4 = (
    -2.7325894832611084, -2.069017171859741, -1.6180464029312134, -1.2562311887741089,
    -0.9423404335975647, -0.6567591428756714, -0.38804829120635986, -0.12839503586292267,
    0.12839503586292267, 0.38804829120635986, 0.6567591428756714, 0.9423404335975647,
    1.2562311887741089, 1.6180464029312134, 2.069017171859741, 2.7325894832611084,
)
# These are binary32-rounded midpoints of the pinned centroids. Index selection
# counts strictly smaller thresholds, so an exact threshold chooses the lower bin.
THRESHOLDS_3 = (
    -1.7479274272918701, -1.049957275390625, -0.5005497336387634, 0.0,
    0.5005497336387634, 1.049957275390625, 1.7479274272918701,
)
THRESHOLDS_4 = (
    -2.400803327560425, -1.843531847000122, -1.4371387958526611, -1.0992858409881592,
    -0.7995498180389404, -0.5224037170410156, -0.25822165608406067, 0.0,
    0.25822165608406067, 0.5224037170410156, 0.7995498180389404, 1.0992858409881592,
    1.4371387958526611, 1.843531847000122, 2.400803327560425,
)


# The64-cell book uses the same analytic Gaussian fixed point and midpoint
# policy. The six-bit wire layout is a64-byte low-nibble plane followed by a
#32-byte high-two-bit plane in each128-component group.
CENTROIDS_6 = (
    -3.7441012859344482, -3.2404370307922363, -2.9174067974090576, -2.672273874282837,
    -2.4713048934936523, -2.298981189727783, -2.146810293197632, -2.009611129760742,
    -1.8839772939682007, -1.7675418853759766, -1.6585888862609863, -1.5558311939239502,
    -1.458276391029358, -1.3651410341262817, -1.275794506072998, -1.1897201538085938,
    -1.1064882278442383, -1.0257363319396973, -0.9471549391746521, -0.8704766035079956,
    -0.7954676747322083, -0.7219219207763672, -0.6496555805206299, -0.5785030126571655,
    -0.508313775062561, -0.43894967436790466, -0.3702826499938965, -0.3021928369998932,
    -0.23456698656082153, -0.167296901345253, -0.10027828812599182, -0.03340950608253479,
    0.03340950608253479, 0.10027828812599182, 0.167296901345253, 0.23456698656082153,
    0.3021928369998932, 0.3702826499938965, 0.43894967436790466, 0.508313775062561,
    0.5785030126571655, 0.6496555805206299, 0.7219219207763672, 0.7954676747322083,
    0.8704766035079956, 0.9471549391746521, 1.0257363319396973, 1.1064882278442383,
    1.1897201538085938, 1.275794506072998, 1.3651410341262817, 1.458276391029358,
    1.5558311939239502, 1.6585888862609863, 1.7675418853759766, 1.8839772939682007,
    2.009611129760742, 2.146810293197632, 2.298981189727783, 2.4713048934936523,
    2.672273874282837, 2.9174067974090576, 3.2404370307922363, 3.7441012859344482,
)
THRESHOLDS_6 = (
    -3.4922690391540527, -3.0789217948913574, -2.7948403358459473, -2.571789264678955,
    -2.3851430416107178, -2.222895622253418, -2.0782108306884766, -1.9467942714691162,
    -1.8257596492767334, -1.7130653858184814, -1.6072100400924683, -1.5070538520812988,
    -1.4117087125778198, -1.3204677104949951, -1.232757329940796, -1.148104190826416,
    -1.0661122798919678, -0.9864456653594971, -0.9088157415390015, -0.8329721689224243,
    -0.7586947679519653, -0.6857887506484985, -0.6140792965888977, -0.5434083938598633,
    -0.47363173961639404, -0.40461617708206177, -0.33623772859573364, -0.26837992668151855,
    -0.20093193650245667, -0.133787602186203, -0.0668438971042633, 0.0,
    0.0668438971042633, 0.133787602186203, 0.20093193650245667, 0.26837992668151855,
    0.33623772859573364, 0.40461617708206177, 0.47363173961639404, 0.5434083938598633,
    0.6140792965888977, 0.6857887506484985, 0.7586947679519653, 0.8329721689224243,
    0.9088157415390015, 0.9864456653594971, 1.0661122798919678, 1.148104190826416,
    1.232757329940796, 1.3204677104949951, 1.4117087125778198, 1.5070538520812988,
    1.6072100400924683, 1.7130653858184814, 1.8257596492767334, 1.9467942714691162,
    2.0782108306884766, 2.222895622253418, 2.3851430416107178, 2.571789264678955,
    2.7948403358459473, 3.0789217948913574, 3.4922690391540527,
)


# Analytic Gaussian Lloyd-Max fixed points solved independently by damped
# Newton iteration; these immutable IEEE754 binary32 tables are the wire
# authority. Midpoints are computed from the pinned centers then rounded once.
CENTROIDS_7 = (
    -4.189693927764893, -3.7349367141723633, -3.447430372238159, -3.231933116912842,
    -3.057246208190918, -2.9090471267700195, -2.7795143127441406, -2.663886547088623,
    -2.5590405464172363, -2.4628114700317383, -2.3736343383789062, -2.290339231491089,
    -2.2120275497436523, -2.137993335723877, -2.067671537399292, -2.0006017684936523,
    -1.9364045858383179, -1.8747625350952148, -1.8154077529907227, -1.7581114768981934,
    -1.7026770114898682, -1.6489338874816895, -1.5967330932617188, -1.5459437370300293,
    -1.4964499473571777, -1.4481489658355713, -1.4009485244750977, -1.3547662496566772,
    -1.3095276355743408, -1.265165090560913, -1.2216174602508545, -1.1788285970687866,
    -1.1367473602294922, -1.0953266620635986, -1.054523229598999, -1.0142966508865356,
    -0.9746099710464478, -0.9354285001754761, -0.8967199325561523, -0.8584542274475098,
    -0.8206029534339905, -0.7831396460533142, -0.7460391521453857, -0.709277868270874,
    -0.6728333234786987, -0.6366842985153198, -0.6008105278015137, -0.5651926398277283,
    -0.529812216758728, -0.4946514666080475, -0.45969343185424805, -0.4249216616153717,
    -0.3903203010559082, -0.35587403178215027, -0.32156795263290405, -0.28738754987716675,
    -0.253318727016449, -0.21934764087200165, -0.18546071648597717, -0.15164464712142944,
    -0.11788628250360489, -0.08417264372110367, -0.050490863621234894, -0.01682817004621029,
    0.01682817004621029, 0.050490863621234894, 0.08417264372110367, 0.11788628250360489,
    0.15164464712142944, 0.18546071648597717, 0.21934764087200165, 0.253318727016449,
    0.28738754987716675, 0.32156795263290405, 0.35587403178215027, 0.3903203010559082,
    0.4249216616153717, 0.45969343185424805, 0.4946514666080475, 0.529812216758728,
    0.5651926398277283, 0.6008105278015137, 0.6366842985153198, 0.6728333234786987,
    0.709277868270874, 0.7460391521453857, 0.7831396460533142, 0.8206029534339905,
    0.8584542274475098, 0.8967199325561523, 0.9354285001754761, 0.9746099710464478,
    1.0142966508865356, 1.054523229598999, 1.0953266620635986, 1.1367473602294922,
    1.1788285970687866, 1.2216174602508545, 1.265165090560913, 1.3095276355743408,
    1.3547662496566772, 1.4009485244750977, 1.4481489658355713, 1.4964499473571777,
    1.5459437370300293, 1.5967330932617188, 1.6489338874816895, 1.7026770114898682,
    1.7581114768981934, 1.8154077529907227, 1.8747625350952148, 1.9364045858383179,
    2.0006017684936523, 2.067671537399292, 2.137993335723877, 2.2120275497436523,
    2.290339231491089, 2.3736343383789062, 2.4628114700317383, 2.5590405464172363,
    2.663886547088623, 2.7795143127441406, 2.9090471267700195, 3.057246208190918,
    3.231933116912842, 3.447430372238159, 3.7349367141723633, 4.189693927764893,
)

THRESHOLDS_7 = (
    -3.962315320968628, -3.591183662414551, -3.339681625366211, -3.14458966255188,
    -2.9831466674804688, -2.84428071975708, -2.721700429916382, -2.6114635467529297,
    -2.5109260082244873, -2.4182229042053223, -2.331986904144287, -2.25118350982666,
    -2.1750104427337646, -2.102832317352295, -2.0341367721557617, -1.9685032367706299,
    -1.9055836200714111, -1.8450851440429688, -1.786759614944458, -1.7303942441940308,
    -1.6758054494857788, -1.622833490371704, -1.571338415145874, -1.5211968421936035,
    -1.4722994565963745, -1.4245487451553345, -1.3778574466705322, -1.3321468830108643,
    -1.287346363067627, -1.2433912754058838, -1.2002229690551758, -1.1577880382537842,
    -1.1160370111465454, -1.0749249458312988, -1.034409999847412, -0.9944533109664917,
    -0.9550192356109619, -0.9160742163658142, -0.877587080001831, -0.8395285606384277,
    -0.8018712997436523, -0.7645894289016724, -0.7276585102081299, -0.6910555958747864,
    -0.6547588109970093, -0.6187474131584167, -0.5830016136169434, -0.5475023984909058,
    -0.5122318267822266, -0.4771724343299866, -0.4423075318336487, -0.40762096643447876,
    -0.37309718132019043, -0.33872097730636597, -0.3044777512550354, -0.27035313844680786,
    -0.2363331913948059, -0.2024041712284088, -0.1685526818037033, -0.13476546108722687,
    -0.10102946311235428, -0.06733175367116928, -0.033659517765045166, 0.0,
    0.033659517765045166, 0.06733175367116928, 0.10102946311235428, 0.13476546108722687,
    0.1685526818037033, 0.2024041712284088, 0.2363331913948059, 0.27035313844680786,
    0.3044777512550354, 0.33872097730636597, 0.37309718132019043, 0.40762096643447876,
    0.4423075318336487, 0.4771724343299866, 0.5122318267822266, 0.5475023984909058,
    0.5830016136169434, 0.6187474131584167, 0.6547588109970093, 0.6910555958747864,
    0.7276585102081299, 0.7645894289016724, 0.8018712997436523, 0.8395285606384277,
    0.877587080001831, 0.9160742163658142, 0.9550192356109619, 0.9944533109664917,
    1.034409999847412, 1.0749249458312988, 1.1160370111465454, 1.1577880382537842,
    1.2002229690551758, 1.2433912754058838, 1.287346363067627, 1.3321468830108643,
    1.3778574466705322, 1.4245487451553345, 1.4722994565963745, 1.5211968421936035,
    1.571338415145874, 1.622833490371704, 1.6758054494857788, 1.7303942441940308,
    1.786759614944458, 1.8450851440429688, 1.9055836200714111, 1.9685032367706299,
    2.0341367721557617, 2.102832317352295, 2.1750104427337646, 2.25118350982666,
    2.331986904144287, 2.4182229042053223, 2.5109260082244873, 2.6114635467529297,
    2.721700429916382, 2.84428071975708, 2.9831466674804688, 3.14458966255188,
    3.339681625366211, 3.591183662414551, 3.962315320968628,
)

CENTROIDS_8 = (
    -4.6035356521606445, -4.186595439910889, -3.925637722015381, -3.731666326522827,
    -3.5755879878997803, -3.4440717697143555, -3.329848527908325, -3.2285001277923584,
    -3.1371326446533203, -3.053741931915283, -2.976882219314575, -2.905472993850708,
    -2.8386857509613037, -2.7758705615997314, -2.71650767326355, -2.6601738929748535,
    -2.6065213680267334, -2.5552594661712646, -2.5061426162719727, -2.4589622020721436,
    -2.413538694381714, -2.3697171211242676, -2.327361822128296, -2.2863545417785645,
    -2.2465898990631104, -2.207975387573242, -2.1704282760620117, -2.1338741779327393,
    -2.0982465744018555, -2.0634851455688477, -2.0295357704162598, -1.99634850025177,
    -1.9638783931732178, -1.9320842027664185, -1.900928020477295, -1.870375156402588,
    -1.840393304824829, -1.810953140258789, -1.7820268869400024, -1.753589391708374,
    -1.7256169319152832, -1.698087453842163, -1.6709803342819214, -1.64427649974823,
    -1.6179578304290771, -1.5920075178146362, -1.5664098262786865, -1.541149616241455,
    -1.5162131786346436, -1.4915870428085327, -1.4672586917877197, -1.4432165622711182,
    -1.4194493293762207, -1.3959463834762573, -1.3726975917816162, -1.3496936559677124,
    -1.326925277709961, -1.3043839931488037, -1.2820614576339722, -1.2599499225616455,
    -1.2380417585372925, -1.2163299322128296, -1.1948076486587524, -1.1734682321548462,
    -1.1523053646087646, -1.1313132047653198, -1.1104860305786133, -1.0898181200027466,
    -1.0693042278289795, -1.0489392280578613, -1.0287182331085205, -1.008636474609375,
    -0.9886893630027771, -0.968872606754303, -0.9491817951202393, -0.929612934589386,
    -0.9101620316505432, -0.8908252716064453, -0.8715988993644714, -0.8524792790412903,
    -0.8334630131721497, -0.8145467042922974, -0.795727014541626, -0.7770008444786072,
    -0.7583650350570679, -0.7398166060447693, -0.7213526368141174, -0.7029702663421631,
    -0.6846667528152466, -0.666439414024353, -0.6482855081558228, -0.63020259141922,
    -0.6121881008148193, -0.5942395925521851, -0.5763546824455261, -0.5585310459136963,
    -0.5407662987709045, -0.523058295249939, -0.5054047703742981, -0.4878036081790924,
    -0.47025266289711, -0.4527498781681061, -0.43529319763183594, -0.417880654335022,
    -0.4005102217197418, -0.3831799626350403, -0.36588799953460693, -0.34863242506980896,
    -0.3314113914966583, -0.314223051071167, -0.2970655858516693, -0.2799372375011444,
    -0.262836217880249, -0.24576078355312347, -0.22870920598506927, -0.21167975664138794,
    -0.19467075169086456, -0.17768049240112305, -0.16070732474327087, -0.14374956488609314,
    -0.12680558860301971, -0.10987371951341629, -0.09295235574245453, -0.0760398581624031,
    -0.059134602546691895, -0.042234983295202255, -0.025339383631944656, -0.008446193300187588,
    0.008446193300187588, 0.025339383631944656, 0.042234983295202255, 0.059134602546691895,
    0.0760398581624031, 0.09295235574245453, 0.10987371951341629, 0.12680558860301971,
    0.14374956488609314, 0.16070732474327087, 0.17768049240112305, 0.19467075169086456,
    0.21167975664138794, 0.22870920598506927, 0.24576078355312347, 0.262836217880249,
    0.2799372375011444, 0.2970655858516693, 0.314223051071167, 0.3314113914966583,
    0.34863242506980896, 0.36588799953460693, 0.3831799626350403, 0.4005102217197418,
    0.417880654335022, 0.43529319763183594, 0.4527498781681061, 0.47025266289711,
    0.4878036081790924, 0.5054047703742981, 0.523058295249939, 0.5407662987709045,
    0.5585310459136963, 0.5763546824455261, 0.5942395925521851, 0.6121881008148193,
    0.63020259141922, 0.6482855081558228, 0.666439414024353, 0.6846667528152466,
    0.7029702663421631, 0.7213526368141174, 0.7398166060447693, 0.7583650350570679,
    0.7770008444786072, 0.795727014541626, 0.8145467042922974, 0.8334630131721497,
    0.8524792790412903, 0.8715988993644714, 0.8908252716064453, 0.9101620316505432,
    0.929612934589386, 0.9491817951202393, 0.968872606754303, 0.9886893630027771,
    1.008636474609375, 1.0287182331085205, 1.0489392280578613, 1.0693042278289795,
    1.0898181200027466, 1.1104860305786133, 1.1313132047653198, 1.1523053646087646,
    1.1734682321548462, 1.1948076486587524, 1.2163299322128296, 1.2380417585372925,
    1.2599499225616455, 1.2820614576339722, 1.3043839931488037, 1.326925277709961,
    1.3496936559677124, 1.3726975917816162, 1.3959463834762573, 1.4194493293762207,
    1.4432165622711182, 1.4672586917877197, 1.4915870428085327, 1.5162131786346436,
    1.541149616241455, 1.5664098262786865, 1.5920075178146362, 1.6179578304290771,
    1.64427649974823, 1.6709803342819214, 1.698087453842163, 1.7256169319152832,
    1.753589391708374, 1.7820268869400024, 1.810953140258789, 1.840393304824829,
    1.870375156402588, 1.900928020477295, 1.9320842027664185, 1.9638783931732178,
    1.99634850025177, 2.0295357704162598, 2.0634851455688477, 2.0982465744018555,
    2.1338741779327393, 2.1704282760620117, 2.207975387573242, 2.2465898990631104,
    2.2863545417785645, 2.327361822128296, 2.3697171211242676, 2.413538694381714,
    2.4589622020721436, 2.5061426162719727, 2.5552594661712646, 2.6065213680267334,
    2.6601738929748535, 2.71650767326355, 2.7758705615997314, 2.8386857509613037,
    2.905472993850708, 2.976882219314575, 3.053741931915283, 3.1371326446533203,
    3.2285001277923584, 3.329848527908325, 3.4440717697143555, 3.5755879878997803,
    3.731666326522827, 3.925637722015381, 4.186595439910889, 4.6035356521606445,
)

THRESHOLDS_8 = (
    -4.3950653076171875, -4.056116580963135, -3.8286519050598145, -3.6536271572113037,
    -3.5098299980163574, -3.386960029602051, -3.279174327850342, -3.182816505432129,
    -3.0954372882843018, -3.0153121948242188, -2.9411776065826416, -2.872079372406006,
    -2.8072781562805176, -2.7461891174316406, -2.688340663909912, -2.633347511291504,
    -2.580890417098999, -2.530701160430908, -2.4825525283813477, -2.4362504482269287,
    -2.391627788543701, -2.348539352416992, -2.3068580627441406, -2.266472339630127,
    -2.2272825241088867, -2.189201831817627, -2.152151107788086, -2.116060256958008,
    -2.0808658599853516, -2.0465104579925537, -2.01294207572937, -1.9801135063171387,
    -1.947981357574463, -1.916506052017212, -1.8856515884399414, -1.8553842306137085,
    -1.825673222541809, -1.796489953994751, -1.767808198928833, -1.7396031618118286,
    -1.7118521928787231, -1.6845338344573975, -1.6576284170150757, -1.6311171054840088,
    -1.604982614517212, -1.5792086124420166, -1.5537797212600708, -1.5286813974380493,
    -1.5039000511169434, -1.4794228076934814, -1.455237627029419, -1.4313329458236694,
    -1.4076979160308838, -1.384321928024292, -1.3611955642700195, -1.3383095264434814,
    -1.3156546354293823, -1.2932226657867432, -1.271005630493164, -1.2489957809448242,
    -1.227185845375061, -1.205568790435791, -1.1841379404067993, -1.1628868579864502,
    -1.1418092250823975, -1.1208996772766113, -1.1001520156860352, -1.0795612335205078,
    -1.0591217279434204, -1.038828730583191, -1.0186773538589478, -0.9986629486083984,
    -0.97878098487854, -0.9590271711349487, -0.9393973350524902, -0.9198874831199646,
    -0.9004936218261719, -0.8812121152877808, -0.8620390892028809, -0.84297114610672,
    -0.8240048885345459, -0.8051368594169617, -0.786363959312439, -0.7676829099655151,
    -0.7490907907485962, -0.7305846214294434, -0.7121614217758179, -0.6938185095787048,
    -0.6755530834197998, -0.6573624610900879, -0.6392440795898438, -0.6211953163146973,
    -0.6032138466835022, -0.5852971076965332, -0.5674428939819336, -0.549648642539978,
    -0.5319123268127441, -0.5142315626144409, -0.49660420417785645, -0.4790281355381012,
    -0.46150127053260803, -0.4440215229988098, -0.42658692598342896, -0.4091954231262207,
    -0.39184510707855225, -0.3745339810848236, -0.35726022720336914, -0.34002190828323364,
    -0.32281720638275146, -0.30564433336257935, -0.28850141167640686, -0.2713867425918579,
    -0.25429850816726685, -0.23723499476909637, -0.2201944887638092, -0.20317524671554565,
    -0.1861756145954132, -0.16919390857219696, -0.152228444814682, -0.13527756929397583,
    -0.1183396577835083, -0.10141304135322571, -0.08449611067771912, -0.0675872266292572,
    -0.050684794783592224, -0.033787183463573456, -0.01689278893172741, 0.0,
    0.01689278893172741, 0.033787183463573456, 0.050684794783592224, 0.0675872266292572,
    0.08449611067771912, 0.10141304135322571, 0.1183396577835083, 0.13527756929397583,
    0.152228444814682, 0.16919390857219696, 0.1861756145954132, 0.20317524671554565,
    0.2201944887638092, 0.23723499476909637, 0.25429850816726685, 0.2713867425918579,
    0.28850141167640686, 0.30564433336257935, 0.32281720638275146, 0.34002190828323364,
    0.35726022720336914, 0.3745339810848236, 0.39184510707855225, 0.4091954231262207,
    0.42658692598342896, 0.4440215229988098, 0.46150127053260803, 0.4790281355381012,
    0.49660420417785645, 0.5142315626144409, 0.5319123268127441, 0.549648642539978,
    0.5674428939819336, 0.5852971076965332, 0.6032138466835022, 0.6211953163146973,
    0.6392440795898438, 0.6573624610900879, 0.6755530834197998, 0.6938185095787048,
    0.7121614217758179, 0.7305846214294434, 0.7490907907485962, 0.7676829099655151,
    0.786363959312439, 0.8051368594169617, 0.8240048885345459, 0.84297114610672,
    0.8620390892028809, 0.8812121152877808, 0.9004936218261719, 0.9198874831199646,
    0.9393973350524902, 0.9590271711349487, 0.97878098487854, 0.9986629486083984,
    1.0186773538589478, 1.038828730583191, 1.0591217279434204, 1.0795612335205078,
    1.1001520156860352, 1.1208996772766113, 1.1418092250823975, 1.1628868579864502,
    1.1841379404067993, 1.205568790435791, 1.227185845375061, 1.2489957809448242,
    1.271005630493164, 1.2932226657867432, 1.3156546354293823, 1.3383095264434814,
    1.3611955642700195, 1.384321928024292, 1.4076979160308838, 1.4313329458236694,
    1.455237627029419, 1.4794228076934814, 1.5039000511169434, 1.5286813974380493,
    1.5537797212600708, 1.5792086124420166, 1.604982614517212, 1.6311171054840088,
    1.6576284170150757, 1.6845338344573975, 1.7118521928787231, 1.7396031618118286,
    1.767808198928833, 1.796489953994751, 1.825673222541809, 1.8553842306137085,
    1.8856515884399414, 1.916506052017212, 1.947981357574463, 1.9801135063171387,
    2.01294207572937, 2.0465104579925537, 2.0808658599853516, 2.116060256958008,
    2.152151107788086, 2.189201831817627, 2.2272825241088867, 2.266472339630127,
    2.3068580627441406, 2.348539352416992, 2.391627788543701, 2.4362504482269287,
    2.4825525283813477, 2.530701160430908, 2.580890417098999, 2.633347511291504,
    2.688340663909912, 2.7461891174316406, 2.8072781562805176, 2.872079372406006,
    2.9411776065826416, 3.0153121948242188, 3.0954372882843018, 3.182816505432129,
    3.279174327850342, 3.386960029602051, 3.5098299980163574, 3.6536271572113037,
    3.8286519050598145, 4.056116580963135, 4.3950653076171875,
)


def _bits(bits: int) -> int:
    if type(bits) is not int or bits not in (3, 4, 6, 7, 8):
        raise ValueError(f"RotorQuant format bits must be 3, 4, 6, 7 or 8, not {bits!r}")
    return bits


# Prepared once with PCG64 seed42. Each bit flips both coordinates of its
# pair, so the diagonal preconditioner has determinant+1. No runtime RNG.
SIGN_PAIR_NEGATIVE_MASK = 0x42F85949E46A0359
WIDE_GIVENS_COS = WIDE_GIVENS_SIN = 0.7071067690849304


def variant_id(variant: str) -> int:
    if variant == "planar":
        return 1
    if variant == "isofast":
        return 2
    if variant == "iso64-norm":
        return 3
    if variant == "signed-iso128":
        return 4
    if variant == "signed-iso64-norm":
        return 5
    if variant == "signed-iso128-norm":
        return 6
    if variant == "signed-iso128-outlier-norm":
        return 7
    if variant == "signed-iso128-norm8":
        return 8
    raise ValueError(f"Unknown RotorQuant reference variant {variant!r}")


def norm_corrected(variant: str) -> bool:
    return variant_id(variant) in (3, 5, 6, 7, 8)


def outlier_scaled(variant: str) -> bool:
    return variant_id(variant) == 7


def centroids(bits: int) -> tuple[float, ...]:
    return {3: CENTROIDS_3, 4: CENTROIDS_4, 6: CENTROIDS_6, 7: CENTROIDS_7, 8: CENTROIDS_8}[_bits(bits)]


def thresholds(bits: int) -> tuple[float, ...]:
    return {3: THRESHOLDS_3, 4: THRESHOLDS_4, 6: THRESHOLDS_6, 7: THRESHOLDS_7, 8: THRESHOLDS_8}[_bits(bits)]


def _variant_bits(bits: int, variant: str) -> int:
    codec = variant_id(variant)
    if codec == 8:
        if bits != 8:
            raise ValueError("Norm8 RotorQuant needs eight-bit indices")
        return codec
    if bits in (7, 8) and codec != 4:
        raise ValueError("Seven/eight-bit RotorQuant needs signed-iso128 original RMS")
    if codec in (6, 7) and bits != 6:
        raise ValueError("Corrected signed-iso128 RotorQuant needs six-bit indices")
    if codec > 2 and bits != 4 and not (codec == 4 and bits in (6, 7, 8)) and not (codec in (6, 7) and bits == 6):
        raise ValueError("Wide RotorQuant needs four-bit indices, or six-bit signed-iso128")
    return codec


def _f32(x: float) -> float:
    try:
        value = struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError as exc:
        raise ValueError("RotorQuant value is outside finite FP32 range") from exc
    if not math.isfinite(value):
        raise ValueError("RotorQuant requires finite FP32 values")
    return value


@dataclass(frozen=True, slots=True)
class FormatDescriptor:
    """Immutable identity for in-process cache copies, prefix reuse and diagnostics.

    Packed bytes do not carry this descriptor. Their owning cache must retain it.
    This version makes no cross-runtime packed-encoding identity claim near an
    FP32 normalization threshold; it fixes table bits, packing and decode meaning.
    """

    variant: str
    bits: int
    codec_id: str
    group: int = GROUP
    scale_dtype: str = "float32"
    norm_policy: str = "original-rms"
    version: int = VERSION
    source_abs_bound: int | None = None
    metadata_max: int | None = None

    @property
    def group_bytes(self) -> int:
        return self.group * self.bits // 8

    @property
    def vector_bytes_per_group(self) -> int:
        return self.group_bytes + 4


def descriptor(bits: int = 4, variant: str = "planar") -> FormatDescriptor:
    """Identity covers all codebook/rotation bits and non-obvious format semantics."""

    bits = _bits(bits)
    codec = _variant_bits(bits, variant)
    if codec == 8:
        # Preserve the independently reviewed private storage UID exactly.
        # Working-basis arithmetic is selected separately from these bytes.
        return FormatDescriptor(variant, bits, NORM8_UID, norm_policy="reconstruction-norm",
                                source_abs_bound=NORM8_SOURCE_BOUND, metadata_max=NORM8_METADATA_BOUND)
    coefficients = PLANAR_COS + PLANAR_SIN if variant == "planar" else ISO_QUATERNION
    header = (f"tensorfold-rotorquant-v{VERSION}|{variant}|bits{bits}|group{GROUP}|"
              "fp32-original-rms|normalize-before-rotation|lower-midpoint|zero-lower-center|"
              "low2-high1-planes" if bits == 3 else
              f"tensorfold-rotorquant-v{VERSION}|{variant}|bits{bits}|group{GROUP}|"
              "fp32-original-rms|normalize-before-rotation|lower-midpoint|zero-lower-center|even-low-nibble")
    if codec > 2:
        header += ("|quaternion-bitpairs01-23-45|" +
                   ("givens-bit6|" if codec in (4, 6, 7, 8) else "") +
                   (f"pair-sign-mask{SIGN_PAIR_NEGATIVE_MASK:016x}|" if codec in (4, 5, 6, 7, 8) else "") +
                   ("fp32-mul-rn-rms-sqrt-rn-div-rn128-centroid-energy|" if norm_corrected(variant) else ""))
        if codec in (4, 6, 7, 8):
            coefficients += (WIDE_GIVENS_COS, WIDE_GIVENS_SIN)
        if outlier_scaled(variant):
            header += "|index-alpha-max1-div-rn-rotated-absmax-outer-centroid|divide-before-index-only"
    if bits == 6:
        header = header.replace("even-low-nibble", "low4-high2-group-local-planes")
    elif bits == 7:
        header = header.replace("even-low-nibble", "low4-mid2-high1-group-local-planes")
    elif bits == 8:
        header = header.replace("even-low-nibble", "group-local-unsigned-byte-indices")
    tables = coefficients + centroids(bits) + thresholds(bits)
    digest = hashlib.sha256(header.encode("ascii") + struct.pack(f"<{len(tables)}f", *tables)).hexdigest()
    return FormatDescriptor(variant, bits, f"rq-v{VERSION}-{variant}{bits}-{digest}",
                            norm_policy="reconstruction-norm" if norm_corrected(variant) else "original-rms")


def _width(width: int) -> None:
    if type(width) is not int or width <= 0 or width % GROUP:
        raise ValueError(f"RotorQuant head dimension must be a positive multiple of {GROUP}, not {width!r}")


def pack_indices_ref(indices: Sequence[int], bits: int = 4) -> bytes:
    """Independent integer packing, each 128-code group's planes stay together.

    3 bits: 32 low-two-bit bytes (four codes each), then 16 high-bit bytes
    (eight codes each). 4 bits:64 bytes, even index in the low nibble.
    6 bits:64 low-nibble bytes followed by32 high-two-bit bytes per group.
    Empty sequences are valid; non-empty tails are rejected rather than padded.
    """

    bits = _bits(bits)
    if len(indices) % GROUP:
        raise ValueError("RotorQuant index count must contain complete 128-component groups")
    maximum = 1 << bits
    for code in indices:
        if type(code) is not int or not 0 <= code < maximum:
            raise ValueError(f"RotorQuant index must be an integer in 0..{maximum - 1}, not {code!r}")
    output = bytearray(len(indices) * bits // 8)
    block_bytes = GROUP * bits // 8
    for group in range(len(indices) // GROUP):
        source, destination = group * GROUP, group * block_bytes
        for i in range(GROUP):
            code = indices[source + i]
            if bits == 3:
                output[destination + i // 4] |= (code & 3) << (2 * (i % 4))
                output[destination + 32 + i // 8] |= (code >> 2) << (i % 8)
            elif bits == 6:
                output[destination + i // 2] |= (code & 15) << (4 * (i % 2))
                output[destination + 64 + i // 4] |= (code >> 4) << (2 * (i % 4))
            elif bits == 7:
                output[destination + i // 2] |= (code & 15) << (4 * (i % 2))
                output[destination + 64 + i // 4] |= ((code >> 4) & 3) << (2 * (i % 4))
                output[destination + 96 + i // 8] |= (code >> 6) << (i % 8)
            elif bits == 8:
                output[destination + i] = code
            else:
                output[destination + i // 2] |= code << (4 * (i % 2))
    return bytes(output)


def unpack_indices_ref(payload: bytes, bits: int = 4) -> list[int]:
    """FP64-oracle decoder for the canonical group-local index planes."""

    bits = _bits(bits)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("RotorQuant payload must be a byte sequence")
    payload = bytes(payload)
    block_bytes = GROUP * bits // 8
    if len(payload) % block_bytes:
        raise ValueError(f"RotorQuant payload must contain complete {block_bytes}-byte groups")
    output = []
    for group in range(len(payload) // block_bytes):
        source = group * block_bytes
        for i in range(GROUP):
            if bits == 3:
                low = (payload[source + i // 4] >> (2 * (i % 4))) & 3
                high = (payload[source + 32 + i // 8] >> (i % 8)) & 1
                output.append(low | (high << 2))
            elif bits == 6:
                low = (payload[source + i // 2] >> (4 * (i % 2))) & 15
                high = (payload[source + 64 + i // 4] >> (2 * (i % 4))) & 3
                output.append(low | (high << 4))
            elif bits == 7:
                low = (payload[source + i // 2] >> (4 * (i % 2))) & 15
                mid = (payload[source + 64 + i // 4] >> (2 * (i % 4))) & 3
                high = (payload[source + 96 + i // 8] >> (i % 8)) & 1
                output.append(low | (mid << 4) | (high << 6))
            elif bits == 8:
                output.append(payload[source + i])
            else:
                output.append((payload[source + i // 2] >> (4 * (i % 2))) & 15)
    return output


def rotate_oracle(values: Sequence[float], variant: str = "planar", *, inverse: bool = False) -> list[float]:
    """Dense FP64 evaluation of pinned transform matrices (inverse is transpose).

    Planar binary32 coefficient rounding introduces ~6e-8 orthogonality error.
    The exact quaternion Isofast table has no coefficient-rounding error.
    """

    codec = variant_id(variant)
    _width(len(values))
    if any(not math.isfinite(x) for x in values):
        raise ValueError("RotorQuant oracle requires finite values")
    if codec > 2:
        return _wide_oracle(values, codec, inverse)
    step = 2 if variant == "planar" else 4
    output = []
    for start in range(0, len(values), step):
        if variant == "planar":
            pair = (start // 2) % len(PLANAR_COS)
            c, s = PLANAR_COS[pair], PLANAR_SIN[pair]
            matrix = ((c, -s), (s, c))
        else:
            w, x, y, z = ISO_QUATERNION
            matrix = ((w, -x, -y, -z), (x, w, -z, y), (y, z, w, -x), (z, -y, x, w))
        if inverse:
            matrix = tuple(zip(*matrix))
        block = values[start:start + step]
        output.extend(math.fsum(coefficient * value for coefficient, value in zip(row, block)) for row in matrix)
    return output


def _wide_oracle(values: Sequence[float], codec: int, inverse: bool) -> list[float]:
    """Independent FP64 matrix evaluation on disjoint bit-pair subspaces."""

    output = list(values)
    width = 128 if codec in (4, 6, 7, 8) else 64
    signed = codec in (4, 5, 6, 7, 8)
    w, x, y, z = ISO_QUATERNION
    matrix = ((w, -x, -y, -z), (x, w, -z, y), (y, z, w, -x), (z, -y, x, w))
    if inverse:
        matrix = tuple(zip(*matrix))

    def signs():
        for i in range(len(output)):
            if SIGN_PAIR_NEGATIVE_MASK >> ((i // 2) % 64) & 1:
                output[i] = -output[i]

    def givens():
        c, s = WIDE_GIVENS_COS, WIDE_GIVENS_SIN
        if inverse:
            s = -s
        for begin in range(0, len(output), 128):
            for i in range(64):
                a, b = output[begin + i], output[begin + i + 64]
                output[begin + i] = math.fsum((c * a, -s * b))
                output[begin + i + 64] = math.fsum((s * a, c * b))

    if signed and not inverse:
        signs()
    if codec in (4, 6, 7, 8) and inverse:
        givens()
    for stride in ((16, 4, 1) if inverse else (1, 4, 16)):
        before = output.copy()
        for begin in range(0, len(output), width):
            for high in range(0, width, 4 * stride):
                for low in range(stride):
                    indices = [begin + high + low + j * stride for j in range(4)]
                    source = [before[i] for i in indices]
                    for i, row in zip(indices, matrix):
                        output[i] = math.fsum(c * v for c, v in zip(row, source))
    if codec in (4, 6, 7, 8) and not inverse:
        givens()
    if signed and inverse:
        signs()
    return output


def quantize_oracle(values: Sequence[float], bits: int = 4, variant: str = "planar") -> tuple[bytes, tuple[float, ...]]:
    """Independent FP64 normalization/rotation, with FP32 metadata rounding.

    Threshold-neighbor codes may differ from the staged FP32 encoder. That is
    expected rounding sensitivity, not a cross-platform packed-bit promise.
    Metadata that overflows FP32 or erases a nonzero group is rejected explicitly.
    """

    bits = _bits(bits)
    _variant_bits(bits, variant)
    _width(len(values))
    if any(not math.isfinite(x) for x in values):
        raise ValueError("RotorQuant oracle requires finite values")
    if variant_id(variant) == 8 and any(abs(x) > NORM8_SOURCE_BOUND for x in values):
        raise ValueError("Norm8 source exceeds the finite 2**30 containment domain")
    boundary = thresholds(bits)
    indices, scales = [], []
    for start in range(0, len(values), GROUP):
        block = values[start:start + GROUP]
        rms = math.hypot(*block) / math.sqrt(GROUP)
        stored = _f32(rms)
        if stored == 0.0 and rms != 0.0:
            raise ValueError("RotorQuant nonzero RMS underflows FP32 metadata")
        normalized = [x / rms for x in block] if rms else [0.0] * GROUP
        rotated = rotate_oracle(normalized, variant)
        if outlier_scaled(variant):
            alpha = max(1.0, max(abs(x) for x in rotated) / centroids(bits)[-1])
            rotated = [x / alpha for x in rotated]
        codes = [bisect.bisect_left(boundary, x) for x in rotated]
        if norm_corrected(variant):
            book = centroids(bits)
            energy = math.fsum(book[code] ** 2 for code in codes)
            stored = _f32(rms * math.sqrt(GROUP / energy))
            if stored == 0.0 and rms != 0.0:
                raise ValueError("RotorQuant nonzero corrected scale underflows FP32 metadata")
            if variant_id(variant) == 8 and stored > NORM8_METADATA_BOUND:
                raise ValueError("Norm8 corrected metadata exceeds 128*2**30")
        indices.extend(codes)
        scales.append(stored)
    return pack_indices_ref(indices, bits), tuple(scales)


def dequantize_oracle(payload: bytes, scales: Sequence[float], bits: int = 4,
                      variant: str = "planar", *, inverse: bool = False) -> list[float]:
    """Independent FP64 reconstruction, optionally returned in original basis."""

    bits = _bits(bits)
    _variant_bits(bits, variant)
    codes = unpack_indices_ref(payload, bits)
    if len(scales) != len(codes) // GROUP:
        raise ValueError("RotorQuant RMS count must match the packed group count")
    if any(not math.isfinite(s) or s < 0 for s in scales):
        raise ValueError("RotorQuant RMS metadata must be finite and nonnegative")
    if variant_id(variant) == 8 and any(s > NORM8_METADATA_BOUND for s in scales):
        raise ValueError("Norm8 corrected metadata exceeds 128*2**30")
    codebook = centroids(bits)
    values = [codebook[code] * scales[i // GROUP] for i, code in enumerate(codes)]
    if inverse and values:
        values = rotate_oracle(values, variant, inverse=True)
    return values


def _wide_rounding_envelope(values: Sequence[float], variant: str,
                            supplied_scales: Sequence[float] | None, *, bits: int = 4) -> tuple[list[float], ...]:
    """Wide normal-domain proof; corrected metadata never becomes raw RMS.

    Gamma127 bounds any legal positive RN sum tree. Three quaternion stages
    propagate absolute coordinate errors with gamma3 signed sums; Givens uses
    gamma2. Corrected scale bounds additionally enclose every allowed centroid
    bin's energy, RN squares/reduction/division/sqrt and final multiplication.
    Supplied scales validate that interval without conditioning normalization.
    Outlier coding first bounds the actual rotated maximum, RN alpha division
    and max(1,alpha), then propagates the second RN coordinate division. The
    stored corrected scale uses raw RMS and centroid energy, never alpha.
    """

    codec = variant_id(variant)
    u, q, tiny, oracle = 2**-24, 2**-150, 2**-126, 64*2**-53
    gamma127 = 127*u/(1-127*u)
    sum_error = (1+u)**3*(1+gamma127)-1
    t_error = max((1+u)*math.sqrt(1+sum_error)-1, 1-(1-u)*math.sqrt(1-sum_error))
    unit_error = (1+u)**2/(1-t_error)-1
    gamma3, gamma2 = 3*u/(1-3*u), 2*u/(1-2*u)
    matrix = ((.5, -.5, -.5, -.5), (.5, .5, -.5, .5),
              (.5, .5, .5, -.5), (.5, -.5, .5, .5))
    coordinates, errors_out, norms, norm_errors = [], [], [], []
    for begin in range(0, len(values), GROUP):
        source = values[begin:begin+GROUP]
        raw_rms = math.hypot(*source)/math.sqrt(GROUP)
        supplied = None if supplied_scales is None else supplied_scales[begin//GROUP]
        if supplied is not None and (supplied < 0 or supplied != _f32(supplied)):
            raise ValueError("RotorQuant rounding envelope scale must be finite nonnegative FP32 metadata")
        if raw_rms == 0:
            if supplied not in (None, 0.):
                raise ValueError("RotorQuant zero source group needs zero scale metadata")
            coordinates.extend([0.]*GROUP)
            errors_out.extend([0.]*GROUP)
            norms.append(0.)
            norm_errors.append(0.)
            continue
        maximum = max(abs(x) for x in source)
        if raw_rms < tiny or any(x != 0 and (abs(x)/maximum)**2 < 2*tiny for x in source):
            raise ValueError("RotorQuant wide rounding envelope needs normal RMS and nonzero squared intermediates")
        raw_error = raw_rms*(t_error+u*(1+t_error)+oracle)+q
        ideal = [x/raw_rms for x in source]
        errors = [unit_error*abs(x)+q*((1+u)*math.sqrt(GROUP)/(1-t_error)+1) for x in ideal]
        if codec in (4, 5, 6, 7, 8):
            ideal = [-x if SIGN_PAIR_NEGATIVE_MASK >> (i//2) & 1 else x for i,x in enumerate(ideal)]
        for low, high in ((0, 1), (2, 3), (4, 5)):
            mask = (1 << low)|(1 << high)
            out, bounds = [0.]*GROUP, [0.]*GROUP
            for base in range(GROUP):
                if base & mask:
                    continue
                indices = (base, base|(1 << low), base|(1 << high), base|mask)
                contribution = .5*math.fsum(abs(ideal[i]) for i in indices)
                propagated = .5*math.fsum(errors[i] for i in indices)
                bound = propagated+gamma3*(contribution+propagated)+4*q/(1-3*u)+oracle*contribution
                bound = math.nextafter(bound*(1+oracle), math.inf)
                for i,row in zip(indices,matrix):
                    out[i] = math.fsum(c*ideal[j] for c,j in zip(row,indices))
                    bounds[i] = bound
            ideal,errors = out,bounds
        if codec in (4, 6, 7, 8):
            out,bounds = [0.]*GROUP,[0.]*GROUP
            for i in range(64):
                a,b = ideal[i],ideal[i+64]
                contribution = WIDE_GIVENS_COS*abs(a)+WIDE_GIVENS_SIN*abs(b)
                propagated = WIDE_GIVENS_COS*errors[i]+WIDE_GIVENS_SIN*errors[i+64]
                bound = propagated+gamma2*(contribution+propagated)+4*q/(1-2*u)+oracle*contribution
                out[i] = math.fsum((WIDE_GIVENS_COS*a,-WIDE_GIVENS_SIN*b))
                out[i+64] = math.fsum((WIDE_GIVENS_SIN*a,WIDE_GIVENS_COS*b))
                bounds[i] = bounds[i+64] = math.nextafter(bound*(1+oracle),math.inf)
            ideal,errors = out,bounds
        if codec == 7:
            endpoint = centroids(bits)[-1]
            alpha = max(1.,max(abs(x) for x in ideal)/endpoint)
            maximum_low = max(max(0.,abs(x)-e) for x,e in zip(ideal,errors))
            maximum_high = max(abs(x)+e for x,e in zip(ideal,errors))
            alpha_low = max(1.,maximum_low/endpoint*(1-u)-q)
            alpha_high = max(1.,maximum_high/endpoint*(1+u)+q)
            alpha_low = max(1.,math.nextafter(alpha_low*(1-oracle),-math.inf))
            alpha_high = math.nextafter(alpha_high*(1+oracle),math.inf)
            out,bounds = [],[]
            for x,e in zip(ideal,errors):
                nominal = x/alpha
                endpoints = ((x-e)/alpha_low,(x-e)/alpha_high,
                             (x+e)/alpha_low,(x+e)/alpha_high)
                magnitude = max(abs(value) for value in endpoints)
                error = max(abs(value-nominal) for value in endpoints)+u*magnitude+q
                error += oracle*(abs(nominal)+magnitude+1)
                out.append(nominal)
                bounds.append(math.nextafter(error*(1+oracle),math.inf))
            ideal,errors = out,bounds
        nominal,lower_scale,upper_scale = raw_rms,raw_rms-raw_error,raw_rms+raw_error
        if norm_corrected(variant):
            boundary,book = thresholds(bits),centroids(bits)
            energy_low,energy_high,energy_nominal = [],[],[]
            for x,e in zip(ideal,errors):
                lower = bisect.bisect_left(boundary,math.nextafter(x-e,-math.inf))
                upper = bisect.bisect_left(boundary,math.nextafter(x+e,math.inf))
                energies = [book[code]**2 for code in range(lower,upper+1)]
                energy_low.append(min(energies))
                energy_high.append(max(energies))
                energy_nominal.append(book[bisect.bisect_left(boundary,x)]**2)
            energy_error = (1+u)*(1+gamma127)-1+oracle
            low = math.fsum(energy_low)*(1-energy_error)
            high = math.fsum(energy_high)*(1+energy_error)
            factor_low = math.sqrt(GROUP/high*(1-u))*(1-u)
            factor_high = math.sqrt(GROUP/low*(1+u))*(1+u)
            nominal *= math.sqrt(GROUP/math.fsum(energy_nominal))
            lower_scale = lower_scale*factor_low*(1-u)-q
            upper_scale = upper_scale*factor_high*(1+u)+q
        lower_scale = math.nextafter(lower_scale*(1-oracle),-math.inf)
        upper_scale = math.nextafter(upper_scale*(1+oracle),math.inf)
        if lower_scale < tiny or upper_scale > 2**128*(1-2**-24):
            raise ValueError("RotorQuant wide rounding envelope needs normal finite metadata")
        if supplied is not None and not lower_scale <= supplied <= upper_scale:
            raise ValueError("RotorQuant supplied scale lies outside the derived wide FP32 rounding bound")
        coordinates.extend(ideal)
        errors_out.extend(errors)
        norms.append(nominal)
        norm_errors.append(math.nextafter(max(nominal-lower_scale,upper_scale-nominal),math.inf))
    return coordinates,errors_out,norms,norm_errors


def rounding_envelope_oracle(values: Sequence[float], variant: str = "planar", *,
                            rms: Sequence[float] | None = None, bits: int = 4) -> tuple[list[float], ...]:
    """Independent FP64 ideal coordinate/RMS values and derived FP32 error bounds.

    Returns ``(rotated, coordinate_abs_error, rms, rms_abs_error)``. This is a
    validation oracle for normal normalization intermediates; zero groups are
    exact. Subnormal scale and underflowing nonzero squared coordinates require
    separate ULP tests and are rejected here, rather than hidden in a tolerance.

    Bound derivation uses binary32 round-to-nearest unit roundoff u=2**-24,
    gamma(n)=n*u/(1-n*u), and the worst depth127 of any 128-term sum tree.
    Normalize division/square contribute (1+u)**3; sqrt and the second divide
    propagate that sum bound. The fixed rotation contributes gamma(2) per
    Planar coordinate or gamma(3) per quaternion signed sum. Bounds scale with
    absolute input contributions, so cancellation remains covered. A separate
    binary64 envelope covers this oracle's hypot, division and bound arithmetic.
    Optional ``rms`` conditions the normalization bound on the actual metadata's
    RN rounding cell: stored_RMS=RN(absmax*t). The ideal RMS must still lie within
    the independently derived global bound. This gives a tighter coordinate
    bound without assuming the runtime's reduction tree or fitting a tolerance.
    This proves admissible bins; it never changes the encoder's tie policy.

    Wide variants propagate all three quaternion stages and optional Givens.
    Their returned norm values/bounds describe encoded metadata (including
    reconstruction norm when enabled). Supplied wide metadata only validates
    its independently derived energy interval; it does not condition the
    original normalization denominator on a corrected stored scale.
    """

    codec = variant_id(variant)
    _variant_bits(_bits(bits),variant)
    _width(len(values))
    for value in values:
        if float(value) != _f32(value):
            raise ValueError("RotorQuant rounding envelope requires exactly represented FP32 source values")
        if codec == 8 and abs(value) > NORM8_SOURCE_BOUND:
            raise ValueError("Norm8 source exceeds the finite 2**30 containment domain")
    if rms is not None and len(rms) != len(values) // GROUP:
        raise ValueError("RotorQuant rounding envelope RMS count must match source groups")
    if codec > 2:
        return _wide_rounding_envelope(values, variant, rms, bits=bits)
    u, q, tiny = 2.0 ** -24, 2.0 ** -150, 2.0 ** -126
    gamma127 = 127 * u / (1 - 127 * u)
    sum_relative_error = (1 + u) ** 3 * (1 + gamma127) - 1
    sqrt_relative_error = max((1 + u) * math.sqrt(1 + sum_relative_error) - 1,
                              1 - (1 - u) * math.sqrt(1 - sum_relative_error))
    rotation_steps = 2 if variant == "planar" else 3
    gamma_rotation = rotation_steps * u / (1 - rotation_steps * u)
    oracle_roundoff = 64 * 2.0 ** -53
    rotated, bounds, norms, norm_bounds = [], [], [], []
    for start in range(0, len(values), GROUP):
        block = values[start:start + GROUP]
        ideal_rms = math.hypot(*block) / math.sqrt(GROUP)
        supplied = None if rms is None else rms[start // GROUP]
        if supplied is not None and (supplied < 0 or supplied != _f32(supplied)):
            raise ValueError("RotorQuant rounding envelope RMS must be finite nonnegative FP32 metadata")
        if ideal_rms == 0:
            if supplied not in (None, 0.0):
                raise ValueError("RotorQuant zero source group needs zero RMS metadata")
            rotated.extend([0.0] * GROUP)
            bounds.extend([0.0] * GROUP)
            norms.append(0.0)
            norm_bounds.append(0.0)
            continue
        maximum = max(abs(value) for value in block)
        if ideal_rms < tiny or any(value != 0 and (abs(value) / maximum) ** 2 < 2 * tiny for value in block):
            raise ValueError("RotorQuant rounding envelope needs normal RMS and nonzero squared intermediates")
        norm_error = ideal_rms * (sqrt_relative_error + u * (1 + sqrt_relative_error) + oracle_roundoff) + q
        norm_error = math.nextafter(norm_error * (1 + oracle_roundoff), math.inf)
        denominator_relative_error = sqrt_relative_error
        if supplied is not None:
            if abs(supplied - ideal_rms) > norm_error:
                raise ValueError("RotorQuant supplied RMS lies outside the derived FP32 rounding bound")
            exponent = (struct.unpack("<I", struct.pack("<f", supplied))[0] >> 23) & 255
            half_ulp = math.ldexp(1.0, exponent - 127 - 24) if exponent else q
            denominator_relative_error = max(abs((supplied - half_ulp) / ideal_rms - 1),
                                               abs((supplied + half_ulp) / ideal_rms - 1)) + oracle_roundoff
        unit_relative_error = (1 + u) ** 2 / (1 - denominator_relative_error) - 1
        unit_subnormal_error = q * ((1 + u) * math.sqrt(GROUP) / (1 - denominator_relative_error) + 1)
        original_rotated = rotate_oracle(block, variant)
        ideal = [value / ideal_rms for value in original_rotated]
        normalized = [value / ideal_rms for value in block]
        step = 2 if variant == "planar" else 4
        for offset in range(0, GROUP, step):
            if variant == "planar":
                pair = offset // 2 % len(PLANAR_COS)
                c, s = abs(PLANAR_COS[pair]), abs(PLANAR_SIN[pair])
                coefficients = ((c, s), (s, c))
            else:
                coefficients = ((0.5,) * 4,) * 4
            absolute_input = [abs(value) for value in normalized[offset:offset + step]]
            input_error = [unit_relative_error * value + unit_subnormal_error for value in absolute_input]
            for row in coefficients:
                propagated = math.fsum(coefficient * error for coefficient, error in zip(row, input_error))
                contribution = math.fsum(coefficient * (value + error)
                                         for coefficient, value, error in zip(row, absolute_input, input_error))
                error = propagated + gamma_rotation * contribution + 4 * q / (1 - rotation_steps * u)
                error += oracle_roundoff * math.fsum(coefficient * value for coefficient, value in zip(row, absolute_input))
                bounds.append(math.nextafter(error * (1 + oracle_roundoff), math.inf))
        rotated.extend(ideal)
        norms.append(ideal_rms)
        norm_bounds.append(norm_error)
    return rotated, bounds, norms, norm_bounds


def index_envelope_oracle(values: Sequence[float], bits: int = 4,
                          variant: str = "planar", *, rms: Sequence[float] | None = None) -> tuple[list[int], list[int]]:
    """Allowed centroid-index interval under the derived FP32 arithmetic bound.

    A singleton interval certifies exact code equality across valid sum trees.
    A two-bin interval is an explicit threshold ambiguity, not an exception that
    permits arbitrary code errors. Exact index selection on a computed threshold
    is still tested independently and must select the lower bin.
    """

    boundary = thresholds(bits)
    _variant_bits(_bits(bits), variant)
    ideal, errors, _, _ = rounding_envelope_oracle(values, variant, rms=rms, bits=bits)
    low = [bisect.bisect_left(boundary, math.nextafter(value - error, -math.inf) if error else value)
           for value, error in zip(ideal, errors)]
    high = [bisect.bisect_left(boundary, math.nextafter(value + error, math.inf) if error else value)
            for value, error in zip(ideal, errors)]
    return low, high


def rotate_ref(x, variant: str = "planar", *, inverse: bool = False):
    """Staged Torch FP32 rotation, preserving leading dimensions."""

    import torch

    codec = variant_id(variant)
    _width(x.shape[-1])
    x = x.to(torch.float32)
    if codec > 2:
        return _wide_ref(x, codec, inverse)
    if variant == "planar":
        pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        selector = torch.arange(x.shape[-1] // 2, device=x.device) % len(PLANAR_COS)
        c = torch.tensor(PLANAR_COS, dtype=torch.float32, device=x.device)[selector]
        s = torch.tensor(PLANAR_SIN, dtype=torch.float32, device=x.device)[selector]
        a, b = pair[..., 0], pair[..., 1]
        output = (torch.stack((c * a + s * b, -s * a + c * b), dim=-1) if inverse else
                  torch.stack((c * a - s * b, s * a + c * b), dim=-1))
    else:
        group = x.reshape(*x.shape[:-1], x.shape[-1] // 4, 4)
        a, b, c, d = (group[..., i] for i in range(4))
        output = (torch.stack((a + b + c + d, -a + b + c - d, -a - b + c + d, -a + b - c + d), dim=-1)
                  if inverse else
                  torch.stack((a - b - c - d, a + b - c + d, a + b + c - d, a - b + c + d), dim=-1)) * 0.5
    return output.reshape(x.shape)


def _wide_ref(x, codec: int, inverse: bool):
    import torch

    shape, width = x.shape, 128 if codec in (4, 6, 7, 8) else 64
    if codec in (4, 5, 6, 7, 8):
        pair = torch.arange(shape[-1], device=x.device) // 2 % 64
        sign = 1 - 2 * ((SIGN_PAIR_NEGATIVE_MASK >> pair) & 1)
        if not inverse:
            x = x * sign
    if codec in (4, 6, 7, 8) and inverse:
        paired = x.reshape(*shape[:-1], shape[-1] // 128, 2, 64)
        a, b = paired[..., 0, :], paired[..., 1, :]
        x = torch.stack((WIDE_GIVENS_COS * a + WIDE_GIVENS_SIN * b,
                         -WIDE_GIVENS_SIN * a + WIDE_GIVENS_COS * b), dim=-2).reshape(shape)
    for stride in ((16, 4, 1) if inverse else (1, 4, 16)):
        block = x.reshape(*shape[:-1], shape[-1] // width, width // (4 * stride), 4, stride)
        a, b, c, d = (block[..., i, :] for i in range(4))
        components = ((a + b + c + d, -a + b + c - d, -a - b + c + d, -a + b - c + d)
                      if inverse else (a - b - c - d, a + b - c + d, a + b + c - d, a - b + c + d))
        x = (torch.stack(components, dim=-2) * 0.5).reshape(shape)
    if codec in (4, 6, 7, 8) and not inverse:
        paired = x.reshape(*shape[:-1], shape[-1] // 128, 2, 64)
        a, b = paired[..., 0, :], paired[..., 1, :]
        x = torch.stack((WIDE_GIVENS_COS * a - WIDE_GIVENS_SIN * b,
                         WIDE_GIVENS_SIN * a + WIDE_GIVENS_COS * b), dim=-2).reshape(shape)
    if codec in (4, 5, 6, 7, 8) and inverse:
        x = x * sign
    return x


def pack_ref(indices, bits: int = 4):
    """Tensorized production-order packing used only for kernel differential tests."""

    import torch

    bits = _bits(bits)
    _width(indices.shape[-1])
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("RotorQuant centroid indices must have an integer dtype")
    if bool(((indices < 0) | (indices >= (1 << bits))).any()):
        raise ValueError("RotorQuant centroid index is outside its bit width")
    groups = indices.to(torch.int32).reshape(*indices.shape[:-1], indices.shape[-1] // GROUP, GROUP)
    if bits == 3:
        low = groups.reshape(*groups.shape[:-1], 32, 4) & 3
        low = sum(low[..., i] << (2 * i) for i in range(4))
        high = groups.reshape(*groups.shape[:-1], 16, 8) >> 2
        high = sum(high[..., i] << i for i in range(8))
        packed = torch.cat((low, high), dim=-1)
    elif bits == 6:
        pair = groups.reshape(*groups.shape[:-1], 64, 2) & 15
        low = pair[..., 0] | (pair[..., 1] << 4)
        high = groups.reshape(*groups.shape[:-1], 32, 4) >> 4
        high = sum(high[..., i] << (2 * i) for i in range(4))
        packed = torch.cat((low, high), dim=-1)
    elif bits == 7:
        pair = groups.reshape(*groups.shape[:-1], 64, 2) & 15
        low = pair[..., 0] | (pair[..., 1] << 4)
        mid = (groups.reshape(*groups.shape[:-1], 32, 4) >> 4) & 3
        mid = sum(mid[..., i] << (2 * i) for i in range(4))
        high = groups.reshape(*groups.shape[:-1], 16, 8) >> 6
        high = sum(high[..., i] << i for i in range(8))
        packed = torch.cat((low, mid, high), dim=-1)
    elif bits == 8:
        packed = groups
    else:
        pair = groups.reshape(*groups.shape[:-1], 64, 2)
        packed = pair[..., 0] | (pair[..., 1] << 4)
    return packed.to(torch.uint8).reshape(*indices.shape[:-1], indices.shape[-1] * bits // 8)


def unpack_ref(payload, bits: int = 4):
    """Tensorized centroid index extraction, separate from the byte-loop oracle."""

    import torch

    bits = _bits(bits)
    size = GROUP * bits // 8
    if payload.dtype != torch.uint8 or payload.shape[-1] <= 0 or payload.shape[-1] % size:
        raise ValueError(f"RotorQuant payload needs uint8 complete {size}-byte groups")
    groups = payload.to(torch.int32).reshape(*payload.shape[:-1], payload.shape[-1] // size, size)
    i = torch.arange(GROUP, device=payload.device)
    if bits == 3:
        codes = ((groups[..., i // 4] >> (2 * (i % 4))) & 3) | (
            ((groups[..., 32 + i // 8] >> (i % 8)) & 1) << 2)
    elif bits == 6:
        codes = ((groups[..., i // 2] >> (4 * (i % 2))) & 15) | (
            ((groups[..., 64 + i // 4] >> (2 * (i % 4))) & 3) << 4)
    elif bits == 7:
        codes = ((groups[..., i // 2] >> (4 * (i % 2))) & 15) | (
            ((groups[..., 64 + i // 4] >> (2 * (i % 4))) & 3) << 4) | (
            ((groups[..., 96 + i // 8] >> (i % 8)) & 1) << 6)
    elif bits == 8:
        codes = groups
    else:
        codes = (groups[..., i // 2] >> (4 * (i % 2))) & 15
    return codes.reshape(*payload.shape[:-1], payload.shape[-1] * 8 // bits)


def quantize_ref(x, bits: int = 4, variant: str = "planar"):
    """Finite floating (...,128*k) input -> uint8 packed indices, original FP32 RMS.

    Stable FP32 norm: a=max(abs(x)), z=x/a, t=sqrt(mean(z*z)), rms=a*t,
    then unit=z/t. Zero denominators become one before division; zero therefore
    deterministically selects the lower center centroid while retaining scale 0.
    """

    import torch

    bits = _bits(bits)
    _variant_bits(bits, variant)
    _width(x.shape[-1])
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("RotorQuant source values must have a supported floating dtype: FP16, BF16 or FP32")
    x = x.to(torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("RotorQuant source values must be finite in FP32")
    if variant_id(variant) == 8 and not bool((x.abs() <= NORM8_SOURCE_BOUND).all()):
        raise ValueError("Norm8 source exceeds the finite 2**30 containment domain")
    groups = x.reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP)
    a = groups.abs().amax(dim=-1)
    z = groups / torch.where(a == 0, torch.ones_like(a), a)[..., None]
    t = (z.square().sum(dim=-1) * (1.0 / GROUP)).sqrt()
    unit = z / torch.where(t == 0, torch.ones_like(t), t)[..., None]
    rms = a * t
    if bool(((a != 0) & (rms == 0)).any()):
        raise ValueError("RotorQuant nonzero RMS underflows FP32 metadata")
    rotated = rotate_ref(unit.reshape(x.shape), variant)
    if outlier_scaled(variant):
        rotated_groups = rotated.reshape(*rms.shape, GROUP)
        alpha = (rotated_groups.abs().amax(dim=-1) / centroids(bits)[-1]).clamp_min(1.0)
        rotated = (rotated_groups / alpha[..., None]).reshape(x.shape)
    boundary = torch.tensor(thresholds(bits), dtype=torch.float32, device=x.device)
    codes = sum((rotated > threshold).to(torch.int32) for threshold in boundary)
    if norm_corrected(variant):
        book = torch.tensor(centroids(bits), dtype=torch.float32, device=x.device)
        energy = book[codes].reshape(*rms.shape, GROUP).square().sum(dim=-1)
        rms = rms * (GROUP / energy).sqrt()
        if not bool(torch.isfinite(rms).all()) or bool(((a != 0) & (rms == 0)).any()):
            raise ValueError("RotorQuant corrected scale exceeds finite nonzero FP32 metadata")
        if variant_id(variant) == 8 and not bool((rms <= NORM8_METADATA_BOUND).all()):
            raise ValueError("Norm8 corrected metadata exceeds 128*2**30")
    return pack_ref(codes, bits), rms


def dequant_ref(payload, rms, bits: int = 4, variant: str = "planar", *, inverse: bool = False):
    """Packed indices and FP32 RMS -> FP32 rotated reconstruction.

    Kernels round the reconstructed K/V working tile to BF16 before dot products.
    This reference leaves that final rounding explicit at its caller.
    """

    import torch

    bits = _bits(bits)
    _variant_bits(bits, variant)
    codes = unpack_ref(payload, bits)
    expected = (*codes.shape[:-1], codes.shape[-1] // GROUP)
    if tuple(rms.shape) != expected or rms.dtype != torch.float32 or rms.device != payload.device:
        raise ValueError("RotorQuant RMS requires matching group shape, FP32 dtype and payload device")
    if not bool((torch.isfinite(rms) & (rms >= 0)).all()):
        raise ValueError("RotorQuant RMS metadata must be finite and nonnegative")
    if variant_id(variant) == 8 and not bool((rms <= NORM8_METADATA_BOUND).all()):
        raise ValueError("Norm8 corrected metadata exceeds 128*2**30")
    book = torch.tensor(centroids(bits), dtype=torch.float32, device=payload.device)
    values = book[codes].reshape(*expected, GROUP) * rms[..., None]
    values = values.reshape(codes.shape)
    if inverse:
        values = rotate_ref(values, variant, inverse=True)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("RotorQuant reconstruction exceeds finite FP32 range")
    return values
