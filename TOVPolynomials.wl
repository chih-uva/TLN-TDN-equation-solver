base = "TOVPolynomials"
 
dir = "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN \
Project/Programs/Equation Solver/"
 
EOS[p_] := ((1 + (4*K*p)^(1/2))^2 - 1)/(4*K)
 
lambdaFromm = -Log[1 - (4*Pi*r^2*(5670*K^3*(K*pc + Sqrt[K*pc]) - 
          (1134*K^2*(2*K*pc + Sqrt[K*pc])*(4*K*pc + Sqrt[K*pc])*
            (1 + 2*Sqrt[K*pc])*Pi*r^2)/Sqrt[K*pc] + 
          27*K*(1 + 4*K*pc + 4*Sqrt[K*pc])*(34*K*pc + 448*K^2*pc^2 + 
            3*Sqrt[K*pc] + 200*(K*pc)^(3/2))*Pi^2*r^4 - (2*K*pc + Sqrt[K*pc])*
           (3 + 580*K*pc + 25632*K^2*pc^2 + 57344*K^3*pc^3 + 24*Sqrt[K*pc] + 
            5552*(K*pc)^(3/2) + 62080*(K*pc)^(5/2))*Pi^3*r^6))/(8505*K^4)]
 
lambdaMatchEq = {\[Lambda]c == 0, True, 
     (-8*Sqrt[K*pc]*(1 + Sqrt[K*pc])*Pi)/(3*K) + \[Lambda]2 == 0, True, 
     (8*Sqrt[K*pc]*(3 + 4*Sqrt[K*pc])*Pi^2)/(45*K^2) + 
       (32*pc*Sqrt[K*pc]*(5 + 7*Sqrt[K*pc])*Pi^2)/(45*K) + \[Lambda]4 == 0, 
     True, (4*Sqrt[K*pc]*(-1 + 22*Sqrt[K*pc])*Pi^3)/(105*K^3) - 
       (128*pc^2*Sqrt[K*pc]*(3 + 28*Sqrt[K*pc])*Pi^3)/(567*K) + 
       (16*pc*Sqrt[K*pc]*(365 + 582*Sqrt[K*pc])*Pi^3)/(2835*K^2) + 
       \[Lambda]6 == 0, True, (4*Sqrt[K*pc]*(5 - 814*Sqrt[K*pc])*Pi^4)/
        (14175*K^4) + (1024*pc^3*Sqrt[K*pc]*(-310 + 203*Sqrt[K*pc])*Pi^4)/
        (42525*K) + (16*pc*Sqrt[K*pc]*(407 + 818*Sqrt[K*pc])*Pi^4)/
        (42525*K^3) - (128*pc^2*Sqrt[K*pc]*(557 + 2576*Sqrt[K*pc])*Pi^4)/
        (42525*K^2) + \[Lambda]8 == 0}
 
lambdaPoly = (4*Pi*r^2*(28350*K^3*(K*pc + Sqrt[K*pc]) - 
       1890*K^2*Sqrt[K*pc]*(3 + 4*Sqrt[K*pc] + 4*K*pc*(5 + 7*Sqrt[K*pc]))*Pi*
        r^2 + 15*K*Sqrt[K*pc]*(27 - 594*Sqrt[K*pc] + 160*K^2*pc^2*
          (3 + 28*Sqrt[K*pc]) - 4*K*pc*(365 + 582*Sqrt[K*pc]))*Pi^2*r^4 - 
       Sqrt[K*pc]*(15 - 2442*Sqrt[K*pc] + 256*K^3*pc^3*
          (-310 + 203*Sqrt[K*pc]) + 4*K*pc*(407 + 818*Sqrt[K*pc]) - 
         32*K^2*pc^2*(557 + 2576*Sqrt[K*pc]))*Pi^3*r^6))/(42525*K^4)
 
lambdaSeries = (-4*Sqrt[K*pc]*Pi*r^2*(3*(5 - 814*Sqrt[K*pc])*Pi^3*r^6 + 
       K*Pi^2*r^4*(405*(-1 + 22*Sqrt[K*pc]) + 4*pc*(407 + 818*Sqrt[K*pc])*Pi*
          r^2) + 2*K^2*Pi*r^2*(945*(3 + 4*Sqrt[K*pc]) + 
         30*pc*(365 + 582*Sqrt[K*pc])*Pi*r^2 - 
         16*pc^2*(557 + 2576*Sqrt[K*pc])*Pi^2*r^4) + 
       K^3*(-28350*(1 + Sqrt[K*pc]) + 7560*pc*(5 + 7*Sqrt[K*pc])*Pi*r^2 - 
         2400*pc^2*(3 + 28*Sqrt[K*pc])*Pi^2*r^4 + 
         256*pc^3*(-310 + 203*Sqrt[K*pc])*Pi^3*r^6)))/(42525*K^4)
 
lambdaSol = {\[Lambda]c -> 0, \[Lambda]2 -> (8*(K*pc + Sqrt[K*pc])*Pi)/(3*K), 
     \[Lambda]4 -> (-8*Sqrt[K*pc]*(3 + 4*Sqrt[K*pc] + 
         4*K*pc*(5 + 7*Sqrt[K*pc]))*Pi^2)/(45*K^2), 
     \[Lambda]6 -> (4*Sqrt[K*pc]*(27 - 594*Sqrt[K*pc] + 
         160*K^2*pc^2*(3 + 28*Sqrt[K*pc]) - 4*K*pc*(365 + 582*Sqrt[K*pc]))*
        Pi^3)/(2835*K^3), \[Lambda]8 -> 
      (-4*Sqrt[K*pc]*(15 - 2442*Sqrt[K*pc] + 256*K^3*pc^3*
          (-310 + 203*Sqrt[K*pc]) + 4*K*pc*(407 + 818*Sqrt[K*pc]) - 
         32*K^2*pc^2*(557 + 2576*Sqrt[K*pc]))*Pi^4)/(42525*K^4)}
 
lamEQ = m[r] == ((1 - E^\[Lambda][r])*r)/2
 
LHSmassEqnPoly = {0, 0, 12*mc*Pi, 0, 20*m2*Pi, 0, 28*m4*Pi, 0, 36*m6*Pi}
 
m2Sub = m2 -> \[Rho]2/5
 
m4Sub = m4 -> \[Rho]4/7
 
m6Sub = m6 -> \[Rho]6/9
 
massEQ = Derivative[1][m][r] == 4*Pi*r^2*\[Rho][r]
 
MCoeffSubFull = {mc -> (K*pc + Sqrt[K*pc])/(3*K), 
     m2 -> -1/15*((2*K*pc + Sqrt[K*pc])*(4*K*pc + Sqrt[K*pc])*
         (1 + 2*Sqrt[K*pc])*Pi)/(K^2*Sqrt[K*pc]), 
     m4 -> ((1 + 4*K*pc + 4*Sqrt[K*pc])*(34*K*pc + 448*K^2*pc^2 + 
         3*Sqrt[K*pc] + 200*(K*pc)^(3/2))*Pi^2)/(630*K^3), 
     m6 -> -1/17010*((2*K*pc + Sqrt[K*pc])*(3 + 580*K*pc + 25632*K^2*pc^2 + 
          57344*K^3*pc^3 + 24*Sqrt[K*pc] + 5552*(K*pc)^(3/2) + 
          62080*(K*pc)^(5/2))*Pi^3)/K^4}
 
mcSub = mc -> \[Rho]c/3
 
mSeries = (2*Pi*r^3*(5670*K^3*(K*pc + Sqrt[K*pc]) - 
       (1134*K^2*(2*K*pc + Sqrt[K*pc])*(4*K*pc + Sqrt[K*pc])*
         (1 + 2*Sqrt[K*pc])*Pi*r^2)/Sqrt[K*pc] + 
       27*K*(1 + 4*K*pc + 4*Sqrt[K*pc])*(34*K*pc + 448*K^2*pc^2 + 
         3*Sqrt[K*pc] + 200*(K*pc)^(3/2))*Pi^2*r^4 - (2*K*pc + Sqrt[K*pc])*
        (3 + 580*K*pc + 25632*K^2*pc^2 + 57344*K^3*pc^3 + 24*Sqrt[K*pc] + 
         5552*(K*pc)^(3/2) + 62080*(K*pc)^(5/2))*Pi^3*r^6))/(8505*K^4)
 
mSub = 4*Pi*r^3*(mc + m2*r^2 + m4*r^4 + m6*r^6)
 
mxFile = "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN \
Project/Programs/Equation Solver/TOVPolynomials.mx"
 
nuEQ = Derivative[1][\[Nu]][r] == (2*(m[r] + 4*Pi*r^3*p[r]))/(r*(r - 2*m[r]))
 
nuLHS = 2*r*\[Nu]2 + 4*r^3*\[Nu]4 + 6*r^5*\[Nu]6 + 8*r^7*\[Nu]8
 
nuMatchEq = {True, (-8*pc*(Pi + 4*Sqrt[K*pc]*Pi))/(3*Sqrt[K*pc]) + 
       2*\[Nu]2 == 0, True, (8*pc*(3 + 14*Sqrt[K*pc])*Pi^2)/
        (45*K*Sqrt[K*pc]) + (64*pc^2*(5 + 16*Sqrt[K*pc])*Pi^2)/
        (45*Sqrt[K*pc]) + 4*\[Nu]4 == 0, True, 
     (4*pc*(-1 + 22*Sqrt[K*pc])*Pi^3)/(105*K^2*Sqrt[K*pc]) - 
       (256*pc^3*(-15 + 112*Sqrt[K*pc])*Pi^3)/(945*Sqrt[K*pc]) + 
       (32*pc^2*(139 + 174*Sqrt[K*pc])*Pi^3)/(945*K*Sqrt[K*pc]) + 6*\[Nu]6 == 
      0, True, (4*pc*(5 - 958*Sqrt[K*pc])*Pi^4)/(14175*K^3*Sqrt[K*pc]) + 
       (2048*pc^4*(-785 + 448*Sqrt[K*pc])*Pi^4)/(42525*Sqrt[K*pc]) - 
       (512*pc^3*(326 + 1103*Sqrt[K*pc])*Pi^4)/(42525*K*Sqrt[K*pc]) + 
       (32*pc^2*(-593 + 2398*Sqrt[K*pc])*Pi^4)/(42525*K^2*Sqrt[K*pc]) + 
       8*\[Nu]8 == 0}
 
nuPoly = (4*(4*K*pc + Sqrt[K*pc])*Pi*r^2)/(3*K) - 
     (2*pc^2*(3 + 14*Sqrt[K*pc] + 8*K*pc*(5 + 16*Sqrt[K*pc]))*Pi^2*r^4)/
      (45*(K*pc)^(3/2)) + (2*Sqrt[K*pc]*(9 - 198*Sqrt[K*pc] + 
        64*K^2*pc^2*(-15 + 112*Sqrt[K*pc]) - 8*K*pc*(139 + 174*Sqrt[K*pc]))*
       Pi^3*r^6)/(2835*K^3) - (Sqrt[K*pc]*(15 - 2874*Sqrt[K*pc] + 
        512*K^3*pc^3*(-785 + 448*Sqrt[K*pc]) - 128*K^2*pc^2*
         (326 + 1103*Sqrt[K*pc]) + 8*K*pc*(-593 + 2398*Sqrt[K*pc]))*Pi^4*r^8)/
      (85050*K^4) + \[Nu]c
 
nuRHS = (4*pc*Pi*r*(192*Sqrt[K*pc]*Pi^4*r^8 + K*Pi^3*r^6*
        (-15*(1 + 154*Sqrt[K*pc]) + pc*(837 + 12550*Sqrt[K*pc])*Pi*r^2) + 
       K^2*Pi^2*r^4*(135*(3 + 158*Sqrt[K*pc]) - 80*pc*(397 + 3340*Sqrt[K*pc])*
          Pi*r^2 + 24*pc^2*(7129 + 49740*Sqrt[K*pc])*Pi^2*r^4) + 
       2*K^3*Pi*r^2*(-2835*(1 + 18*Sqrt[K*pc]) + 
         540*pc*(201 + 922*Sqrt[K*pc])*Pi*r^2 - 
         320*pc^2*(2249 + 7136*Sqrt[K*pc])*Pi^2*r^4 + 
         192*pc^3*(13451 + 35691*Sqrt[K*pc])*Pi^3*r^6) + 
       2*K^4*(14175*(1 + 4*Sqrt[K*pc]) - 45360*pc*(5 + 6*Sqrt[K*pc])*Pi*r^2 + 
         4320*pc^2*(261 + 224*Sqrt[K*pc])*Pi^2*r^4 - 
         40960*pc^3*(97 + 70*Sqrt[K*pc])*Pi^3*r^6 + 
         4352*pc^4*(2385 + 1568*Sqrt[K*pc])*Pi^4*r^8)))/
     (5*K*(12*pc*(1 + 10*Sqrt[K*pc])*Pi^4*r^8 + 4*K*pc*Pi^3*r^6*
        (-27*(3 + 46*Sqrt[K*pc]) + 4*pc*(157 + 1678*Sqrt[K*pc])*Pi*r^2) + 
       8*K^2*pc*Pi^2*r^4*(567*(1 + 8*Sqrt[K*pc]) - 
         54*pc*(87 + 346*Sqrt[K*pc])*Pi*r^2 + 224*pc^2*(82 + 253*Sqrt[K*pc])*
          Pi^2*r^4) + K^3*(8505*Sqrt[K*pc] - 22680*pc*(1 + Sqrt[K*pc])*Pi*
          r^2 + 18144*pc^2*(5 + 4*Sqrt[K*pc])*Pi^2*r^4 - 
         3456*pc^3*(81 + 56*Sqrt[K*pc])*Pi^3*r^6 + 
         1024*pc^4*(709 + 448*Sqrt[K*pc])*Pi^4*r^8)))
 
nuSeriesEq = (2*(6*pc*(5 - 958*Sqrt[K*pc])*Pi^4*r^7 + 
       2*K*pc*Pi^3*r^5*(405*(-1 + 22*Sqrt[K*pc]) + 
         8*pc*(-593 + 2398*Sqrt[K*pc])*Pi*r^2) - 4*K^2*pc*Pi^2*r^3*
        (-945*(3 + 14*Sqrt[K*pc]) - 180*pc*(139 + 174*Sqrt[K*pc])*Pi*r^2 + 
         64*pc^2*(326 + 1103*Sqrt[K*pc])*Pi^2*r^4) + 
       K^3*r*(-56700*pc*(Pi + 4*Sqrt[K*pc]*Pi) + 30240*pc^2*
          (5 + 16*Sqrt[K*pc])*Pi^2*r^2 - 5760*pc^3*(-15 + 112*Sqrt[K*pc])*
          Pi^3*r^4 + 1024*pc^4*(-785 + 448*Sqrt[K*pc])*Pi^4*r^6 + 
         42525*Sqrt[K*pc]*(\[Nu]2 + 2*r^2*\[Nu]4 + 3*r^4*\[Nu]6 + 
           4*r^6*\[Nu]8))))/(42525*K^3*Sqrt[K*pc])
 
nuSol = {\[Nu]2 -> (4*(4*K*pc + Sqrt[K*pc])*Pi)/(3*K), 
     \[Nu]4 -> (-2*pc^2*(3 + 14*Sqrt[K*pc] + 8*K*pc*(5 + 16*Sqrt[K*pc]))*
        Pi^2)/(45*(K*pc)^(3/2)), \[Nu]6 -> 
      (2*Sqrt[K*pc]*(9 - 198*Sqrt[K*pc] + 64*K^2*pc^2*
          (-15 + 112*Sqrt[K*pc]) - 8*K*pc*(139 + 174*Sqrt[K*pc]))*Pi^3)/
       (2835*K^3), \[Nu]8 -> -1/85050*(Sqrt[K*pc]*(15 - 2874*Sqrt[K*pc] + 
          512*K^3*pc^3*(-785 + 448*Sqrt[K*pc]) - 128*K^2*pc^2*
           (326 + 1103*Sqrt[K*pc]) + 8*K*pc*(-593 + 2398*Sqrt[K*pc]))*Pi^4)/
        K^4}
 
p2Sub = p2 -> (-2*Pi*(pc + \[Rho]c)*(3*pc + \[Rho]c))/3
 
p4Sub = p4 -> (2*Pi*(30*pc^3*Pi - 9*pc*\[Rho]2 + 40*pc^2*Pi*\[Rho]c - 
        4*\[Rho]2*\[Rho]c + 10*pc*Pi*\[Rho]c^2))/15
 
p6Sub = p6 -> (-2*Pi*(3780*pc^4*Pi^2 - 1008*pc^2*Pi*\[Rho]2 + 63*\[Rho]2^2 + 
        360*pc*\[Rho]4 + 5040*pc^3*Pi^2*\[Rho]c - 168*pc*Pi*\[Rho]2*\[Rho]c + 
        150*\[Rho]4*\[Rho]c + 1680*pc^2*Pi^2*\[Rho]c^2 + 
        140*Pi*\[Rho]2*\[Rho]c^2 + 560*pc*Pi^2*\[Rho]c^3 + 
        140*Pi^2*\[Rho]c^4))/945
 
p8Sub = p8 -> (Pi*(226800*pc^5*Pi^3 - 52920*pc^3*Pi^2*\[Rho]2 - 
        1512*pc*Pi*\[Rho]2^2 + 18900*pc^2*Pi*\[Rho]4 - 2430*\[Rho]2*\[Rho]4 - 
        7875*pc*\[Rho]6 + 302400*pc^4*Pi^3*\[Rho]c - 10080*pc^2*Pi^2*\[Rho]2*
         \[Rho]c - 4032*Pi*\[Rho]2^2*\[Rho]c + 900*pc*Pi*\[Rho]4*\[Rho]c - 
        3150*\[Rho]6*\[Rho]c + 126000*pc^3*Pi^3*\[Rho]c^2 - 
        16800*pc*Pi^2*\[Rho]2*\[Rho]c^2 - 3300*Pi*\[Rho]4*\[Rho]c^2 + 
        50400*pc^2*Pi^3*\[Rho]c^3 - 10640*Pi^2*\[Rho]2*\[Rho]c^3 - 
        5600*pc*Pi^3*\[Rho]c^4 - 5600*Pi^3*\[Rho]c^5))/14175
 
pSeries = (pc*(3*(64 + 279*Sqrt[K*pc])*Pi^4*r^8 + 
       2*K*Pi^3*r^6*(-270*(4 + 53*Sqrt[K*pc]) + pc*(6275 + 85548*Sqrt[K*pc])*
          Pi*r^2) + 12*K^2*Pi^2*r^4*(315*(4 + 45*Sqrt[K*pc]) - 
         30*pc*(649 + 3488*Sqrt[K*pc])*Pi*r^2 + 
         8*pc^2*(12435 + 53804*Sqrt[K*pc])*Pi^2*r^4) + 
       36*K^3*Pi*r^2*(-1575*(1 + 6*Sqrt[K*pc]) + 
         210*pc*(107 + 252*Sqrt[K*pc])*Pi*r^2 - 
         80*pc^2*(1389 + 2444*Sqrt[K*pc])*Pi^2*r^4 + 
         32*pc^3*(11897 + 18020*Sqrt[K*pc])*Pi^3*r^6) + 
       14*K^4*(6075 - 32400*pc*Pi*r^2 + 120960*pc^2*Pi^2*r^4 - 
         368640*pc^3*Pi^3*r^6 + 974848*pc^4*Pi^4*r^8)))/(85050*K^4)
 
pSub = pc + p2*r^2 + p4*r^4 + p6*r^6 + p8*r^8
 
rhoSeries = pc + pc/Sqrt[K*pc] - ((2*K*pc + Sqrt[K*pc])*(4*K*pc + Sqrt[K*pc])*
       (1 + 2*Sqrt[K*pc])*Pi*r^2)/(3*K^2*Sqrt[K*pc]) + 
     ((1 + 4*K*pc + 4*Sqrt[K*pc])*(34*K*pc + 448*K^2*pc^2 + 3*Sqrt[K*pc] + 
        200*(K*pc)^(3/2))*Pi^2*r^4)/(90*K^3) - 
     ((2*K*pc + Sqrt[K*pc])*(3 + 580*K*pc + 25632*K^2*pc^2 + 57344*K^3*pc^3 + 
        24*Sqrt[K*pc] + 5552*(K*pc)^(3/2) + 62080*(K*pc)^(5/2))*Pi^3*r^6)/
      (1890*K^4) + ((54591488*K^5*pc^5 + 15*Sqrt[K*pc] - 
        2*K*pc*(669 + 3050*Sqrt[K*pc]) + 8704*K^4*pc^4*
         (7215 + 10492*Sqrt[K*pc]) + 384*K^3*pc^3*
         (13745 + 61476*Sqrt[K*pc]) + 8*K^2*pc^2*(4313 + 83508*Sqrt[K*pc]))*
       Pi^4*r^8)/(340200*K^5)
 
RHSmassEqnPoly = {0, 0, 4*Pi*\[Rho]c, 0, 4*Pi*\[Rho]2, 0, 4*Pi*\[Rho]4, 0, 
     4*Pi*\[Rho]6, 0, 4*Pi*\[Rho]8}
 
TOV = Derivative[1][p][r] == ((m[r] + 4*Pi*r^3*p[r])*(-p[r] - \[Rho][r]))/
      (r*(r - 2*m[r]))
 
TOVPoly = 2*r*(p2 + 2*p4*r^2 + 3*p6*r^4 + 4*p8*r^6) == 
     (4*Pi*r*(pc + p2*r^2 + p4*r^4 + p6*r^6 + p8*r^8 + r^2*\[Rho]2 + 
        r^4*\[Rho]4 + r^6*\[Rho]6 + r^8*\[Rho]8 + \[Rho]c)*
       (315*pc + 315*p2*r^2 + 315*p4*r^4 + 315*p6*r^6 + 315*p8*r^8 + 
        63*r^2*\[Rho]2 + 45*r^4*\[Rho]4 + 35*r^6*\[Rho]6 + 105*\[Rho]c))/
      (-315 + 8*Pi*(63*r^4*\[Rho]2 + 45*r^6*\[Rho]4 + 35*r^8*\[Rho]6 + 
         105*r^2*\[Rho]c))
 
wlFile = "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN \
Project/Programs/Equation Solver/TOVPolynomials.wl"
 
\[Lambda]Sub = r^2*\[Lambda]2 + r^4*\[Lambda]4 + r^6*\[Lambda]6 + 
     r^8*\[Lambda]8 + \[Lambda]c
 
\[Nu]Sub = r^2*\[Nu]2 + r^4*\[Nu]4 + r^6*\[Nu]6 + r^8*\[Nu]8 + \[Nu]c
 
\[Rho]2Sub = {\[Rho]2 -> (p2*Sqrt[K*pc]*(1 + 2*Sqrt[K*pc]))/(2*K*pc)}
 
\[Rho]2Subpc = {\[Rho]2 -> -1/3*(Sqrt[K*pc]*(1 + 2*Sqrt[K*pc])*
         (pc + (K*pc + Sqrt[K*pc])/K)*(3*pc + (K*pc + Sqrt[K*pc])/K)*Pi)/
        (K*pc)}
 
\[Rho]4Sub = {\[Rho]4 -> (8*K*p4*pc^2 - p2^2*Sqrt[K*pc] + 4*p4*pc*Sqrt[K*pc])/
       (8*K*pc^2)}
 
\[Rho]4Subpc = {\[Rho]4 -> ((1 + 4*K*pc + 4*Sqrt[K*pc])*
        (34*K*pc + 448*K^2*pc^2 + 3*Sqrt[K*pc] + 200*(K*pc)^(3/2))*Pi^2)/
       (90*K^3)}
 
\[Rho]6Sub = {\[Rho]6 -> (16*K*p6*pc^3 + p2^3*Sqrt[K*pc] - 
        4*p2*p4*pc*Sqrt[K*pc] + 8*p6*pc^2*Sqrt[K*pc])/(16*K*pc^3)}
 
\[Rho]6Subpc = {\[Rho]6 -> -1/1890*((2*K*pc + Sqrt[K*pc])*
         (3 + 580*K*pc + 25632*K^2*pc^2 + 57344*K^3*pc^3 + 24*Sqrt[K*pc] + 
          5552*(K*pc)^(3/2) + 62080*(K*pc)^(5/2))*Pi^3)/K^4}
 
\[Rho]8Sub = {\[Rho]8 -> (128*K*p8*pc^4 - 5*p2^4*Sqrt[K*pc] + 
        24*p2^2*p4*pc*Sqrt[K*pc] - 16*p4^2*pc^2*Sqrt[K*pc] - 
        32*p2*p6*pc^2*Sqrt[K*pc] + 64*p8*pc^3*Sqrt[K*pc])/(128*K*pc^4)}
 
\[Rho]8Subpc = {\[Rho]8 -> ((54591488*K^5*pc^5 + 15*Sqrt[K*pc] - 
         2*K*pc*(669 + 3050*Sqrt[K*pc]) + 8704*K^4*pc^4*
          (7215 + 10492*Sqrt[K*pc]) + 384*K^3*pc^3*
          (13745 + 61476*Sqrt[K*pc]) + 8*K^2*pc^2*(4313 + 83508*Sqrt[K*pc]))*
        Pi^4)/(340200*K^5)}
 
\[Rho]cSub = {\[Rho]c -> (K*pc + Sqrt[K*pc])/K}
 
\[Rho]EOS = (K*pc + Sqrt[K*pc])/K + (p2*Sqrt[K*pc]*(1 + 2*Sqrt[K*pc])*r^2)/
      (2*K*pc) + (((K*p2^2)/pc + (-1/2*p2^2/pc^2 + (2*p4)/pc)*Sqrt[K*pc]*
         (1 + 2*Sqrt[K*pc]))*r^4)/(4*K) + 
     ((K*p2*(-1/2*p2^2/pc^2 + (2*p4)/pc) + 
        (2*((3*p6)/pc - (3*p2*(-1/2*p2^2/pc^2 + (2*p4)/pc))/(4*pc))*
          Sqrt[K*pc]*(1 + 2*Sqrt[K*pc]))/3)*r^6)/(4*K) + 
     (((2*K*p2*((3*p6)/pc - (3*p2*(-1/2*p2^2/pc^2 + (2*p4)/pc))/(4*pc)))/3 + 
        (K*(-1/2*p2^2/pc^2 + (2*p4)/pc)^2*pc)/4 + 
        (((p2*p6)/(2*pc^2) + (4*p8)/pc - (p4*(-1/2*p2^2/pc^2 + (2*p4)/pc))/
            (2*pc) - (5*p2*((3*p6)/pc - (3*p2*(-1/2*p2^2/pc^2 + (2*p4)/pc))/(
                4*pc)))/(6*pc))*Sqrt[K*pc]*(1 + 2*Sqrt[K*pc]))/2)*r^8)/(4*K)
 
\[Rho]Sub = r^2*\[Rho]2 + r^4*\[Rho]4 + r^6*\[Rho]6 + r^8*\[Rho]8 + \[Rho]c
