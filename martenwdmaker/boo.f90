      program boo

      implicit none

      double precision dens, dens2
      dens = 2.053213937959257D17
      dens2 = 2.05321393D17
      write(6,*) 'number =',dens
 01   format(1x,5(a,1pe16.8))
      write(6,*) 'number =', dens2

      end

