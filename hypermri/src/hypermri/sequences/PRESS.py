import logging
from ..utils.utils_logging import LOG_MODES, init_default_logger
from ..brukerexp import BrukerExp
from .base_spectroscopy import BaseSpectroscopy
import numpy as np
from ..utils.utils_spectroscopy import get_freq_axis

import hypermri.utils.utils_fitting as utf

import hypermri.utils.utils_spectroscopy as uts
import matplotlib.pyplot as plt
from ipywidgets import widgets
from matplotlib.patches import Rectangle
# initialize logger
logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


class PRESS(BaseSpectroscopy):
    def __init__(self, path_or_BrukerExp, *args, **kwargs):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path
        tmp_brukerexp = BrukerExp(path_or_BrukerExp, load_data=False)
        super().__init__(path_or_BrukerExp, *args, **kwargs)
        self.Nvox = self.method["PVM_NVoxels"]
        self.NR = self.method["PVM_NRepetitions"]
        self.TR = self.method['PVM_RepetitionTime'] / 1000
        self.complex_spec,self.complex_fids = self.get_fids_spectra(0,0)
        self.ppm_axis = self.get_ppm(0)
        if self.NR>1:
            if self.n_receivers==1:
                self.time_ax_array = np.array([np.arange(n, self.NR * self.TR * self.Nvox, self.TR * self.Nvox) for n in range(self.Nvox)])  # in seconds
            else:
                self.time_ax_array=self.time_axis
                logger.warning('Caution, self.time_axis might not be correct for dynamic MV-PRESS here')
        if tmp_brukerexp.n_receivers == 2:
            logger.debug('This is dual channel data, performing phasing and adding phased spectra and fids.')
            phase = self.find_phase_shift_dual_channel(0,0,False)
            self.phased_complex_spec,self.phased_complex_fid = self.apply_phase_shift_dual_channel(phase,0,0)
        else:
            self.phased_complex_spec, self.phased_complex_fid = np.zeros_like(self.complex_spec),np.zeros_like(self.complex_fids)
    def get_ppm(self,cut_off=0):
        return get_freq_axis(scan=self, unit="ppm", cut_off=cut_off, npoints=None)
    def get_hz(self,cut_off=0):
        return get_freq_axis(scan=self, unit="Hz", cut_off=cut_off, npoints=None)
    def get_fids_spectra(
            self,LB=0, cut_off=0
    ):
        """Retrieves fids and spectra from rawdatajob0 file and puts it into format
        (spectral_points,voxels, 1,1, reps, channels).
        dimensions 2 and 3 are set to 1 since here we have no read/phase spatial encoding
        Parameters
        ----------
        LB: float, optional
            linebroadening applied in Hz, default is 0.c
        cut_off: int, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        complex_spec: array
            complex spectra in 6D array format
        fids: array
            complex fids sorted into 6D array format
        """
        if len(self.rawdatajob0) == 0:
           data = self.fid
        else:
           data = self.rawdatajob0
        ac_points = self.method["PVM_SpecMatrix"]
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        Nvox = self.method['PVM_NVoxels']
        Nchan = self.n_receivers
        time_ax = np.linspace(0, ac_time, ac_points)[cut_off:] / 1000
        sigma = 2.0 * np.pi * LB
        complex_spec=np.zeros((ac_points-cut_off,self.Nvox,1,1,self.NR,Nchan),dtype=complex)
        complex_fids = np.zeros((ac_points - cut_off, self.Nvox, 1, 1, self.NR, Nchan),dtype=complex)

        for repetition in range(self.NR):
            for voxel in range(Nvox):
                for channel in range(Nchan):
                    # TODO this bug should be fixed now
                    start_idx = cut_off + (repetition * Nvox + voxel + channel) * ac_points
                    fid = data[start_idx:start_idx+ac_points-cut_off]
                    #print('startidx',start_idx,'Rep',repetition,'Vox',voxel,'Chan',channel,'FID size',fid.shape,'time_ax size',time_ax.shape)
                    lb_fid = fid * np.exp(-sigma * time_ax)
                    complex_spec[:,voxel,0,0,repetition,channel] = np.fft.fftshift(np.fft.fft(lb_fid))
                    complex_fids[:,voxel,0,0,repetition,channel] = lb_fid
        return complex_spec,complex_fids
    def plot(self,LB=5,cut_off=0):
        """
        Plots spectra from MV-PRESS dataset.


        Returns
        -------
        """
        # TODO implement phasing of channels in plot as well as phase correction of real spectra
        fig,ax=plt.subplots(1,figsize=(8,4))
        complex_spec,_ = self.get_fids_spectra(LB,cut_off)
        ppm = self.get_ppm(cut_off)
        if self.n_receivers==2:
            phase = self.find_phase_shift_dual_channel(LB, cut_off, False)
            phased_complex_spec, _ = self.apply_phase_shift_dual_channel(phase, LB,cut_off)
        else:
            phased_complex_spec=np.zeros_like(complex_spec)
        @widgets.interact(voxel=(0,self.Nvox-1,1),repetition=(0,self.NR-1,1),channel=(0,self.n_receivers-1,1))
        def update(voxel=0,repetition=0,channel=0):
            ax.cla()

            #ax.plot(ppm,np.real(complex_spec[:,voxel,0,0,repetition,channel]),color='C0',label='Real')
            ax.plot(ppm,np.abs(complex_spec[:, voxel, 0, 0, repetition, channel]),label='Mag',color='k')
            if self.n_receivers==2:
                ax.plot(ppm,np.abs(phased_complex_spec[:,voxel,0,0,repetition,0]),color='C1',label='2 Chan. phased')
            ax.legend()
            ax.set_title('Chan '+str(channel)+', '+str(LB) + ' Hz')
            ax.set_xlim([np.max(ppm)+1,np.min(ppm)-1])
            ax.set_xlabel(r'$\sigma [ppm]$')
            ax.set_ylabel('I [a.u.]')
    def find_phase_shift_dual_channel(
        self, LB=0, cut_off=0, plot=False
    ):
        """
        Finds the phase shift between data from a dual channel coil

        Parameters
        -------
        lb: float, optional
            Linebroadening applied to spectra in Hz.
        cut_off: int, optional
            Number of points that are left out for recorded fid as they are just noise at the beginning, default is 70.
        plot: bool, optional
            if a plot of the result is wanted for QA, can be turned to True

        Returns
        -------
        final_phase: float, phase in degree that maximizes integral of both channels
        """


        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
        fid_ch1 = self.complex_fids[cut_off:, :, :, :, :, 0]
        fid_ch2 = self.complex_fids[cut_off:, :, :, :, :, 1]

        # finding out which of the two channels has the highest signal at which voxel summed over
        # all repetitions so that we phase the voxel that has the most signal
        # first integrate the signal for each of the two channels seperately
        integral_ch1 = np.zeros(((self.Nvox,self.NR)))
        integral_ch2 = np.zeros(((self.Nvox,self.NR)))

        for v in range(self.Nvox):
            for r in range(self.NR):
                integral_ch1[v,r]=np.sum(np.abs(fid_ch1[:, v, 0, 0, r]))
                integral_ch2[v,r]=np.sum(np.abs(fid_ch2[:, v, 0, 0, r]))

        # find out at which voxel the signal is maximized
        max_signal_ch1 = np.squeeze(np.where(np.abs(integral_ch1 - np.max(integral_ch1)) == 0))
        max_signal_ch2 = np.squeeze(np.where(np.abs(integral_ch2 - np.max(integral_ch2)) == 0))

        # check which channel has the larger difference (i.e. more signal)
        # we cant just compare the max values cause the background offset is different
        # i.e. channel 2 could have the same absolute intensity maximum but relatively
        # it has lower signal
        signal_diff_ch1 = np.max(integral_ch1) - np.min(integral_ch1)
        signal_diff_ch2 = np.max(integral_ch2) - np.min(integral_ch2)
        # selecting which index to use
        if signal_diff_ch1 > signal_diff_ch2:
            max_signal_vox = max_signal_ch1
        else:
            max_signal_vox = max_signal_ch2
        ch_1 = fid_ch1[:,max_signal_vox[0],0,0, max_signal_vox[1]]
        ch_2 = fid_ch2[:, max_signal_vox[0], 0,0,max_signal_vox[1]]
        # Can apply linebroadening here for nicer plots
        sigma = 2 * np.pi * LB
        ch_1_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_1 * np.exp(-sigma * time_ax))))
        ch_2_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_2 * np.exp(-sigma * time_ax))))

        ppm = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off, npoints=None)

        # finding optimal phase shift between channels
        Integrals = []
        phases = np.linspace(0, 360, 1000)
        for phase in phases:
            itgl = np.sum(np.abs(ch_1 * np.exp(1j * (phase * np.pi) / 180.0) + ch_2))
            Integrals.append(itgl)

        final_phase = phases[np.argmin(np.abs(Integrals - np.max(Integrals)))]

        # Optional plotting for QA
        if plot is True:
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(phases, Integrals / np.max(Integrals))
            ax.set_xlabel(r"$\phi$ [rad]")
            ax.set_ylabel("Integral of ch_1 phaseshifted against ch_2")
            ax.vlines(
                final_phase,
                np.min(Integrals / np.max(Integrals)),
                1,
                color="orange",
            )
            ax.set_title(r"$\phi$ = " + str(np.round(final_phase, 1)) + "deg")

            # now plot spectra
            best_spec = np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        (ch_1 * np.exp(1j * (final_phase * np.pi) / 180.0) + ch_2)
                        * np.exp(-sigma * time_ax)
                    )
                )
            )

            ax2.plot(ppm, ch_1_spec / np.max(best_spec), label="Ch_1 spec")
            ax2.plot(ppm, ch_2_spec / np.max(best_spec), label="Ch_2 spec")
            ax2.plot(
                ppm,
                best_spec / np.max(best_spec),
                label="Both Channels spec",
            )
            ax2.set_xlabel(r"$\sigma$[ppm]")
            ax2.set_ylabel("I[a.u.]")
            ax2.legend(loc="best", ncol=1)
            ax2.set_title("Spectra from dual channel data")
            ax2.set_xlim([np.max(ppm), np.min(ppm)])
            minimum = np.argmin(np.abs(Integrals - np.min(Integrals)))
            fig.suptitle('Highest signal voxel: Num'+str(max_signal_vox[0])+', Rep:'+str(max_signal_vox[1]))
            plt.tight_layout()
        else:
            pass

        return final_phase

    def apply_phase_shift_dual_channel(
            self,phase_shift,LB=0,cut_off=0):
        """
        Applies a phase shift to combine signal from two channels for maximum intensity.


        Returns
        -------
        complex_spec_phased: 6D nd.array (spectral_points, NumVoxels, 1,1,NumRepetitons,1)
        """
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        time_ax = np.linspace(0, ac_time, ac_points-cut_off) / 1000
        sigma = 2 * np.pi * LB
        ch_1 = self.complex_fids[cut_off:, :, :, :, :, 0]
        ch_2 = self.complex_fids[cut_off:, :, :, :, :, 1]
        fid_phased = np.zeros((ac_points - cut_off, self.Nvox, 1, 1, self.NR, 1),dtype=complex)
        complex_spec_phased = np.zeros((ac_points - cut_off, self.Nvox, 1, 1, self.NR, 1), dtype=complex)
        for r in range(self.NR):
            for n in range(self.Nvox):
                fid_temp = ch_1[:,n, 0,0,r] * np.exp(1j * (phase_shift * np.pi) / 180.0) + ch_2[:,n, 0,0,r]
                fid_phased[:,n, 0,0,r,0] = fid_temp
                complex_spec_phased[:, n, 0, 0, r, 0] = np.fft.fftshift(np.fft.fft(fid_temp*np.exp(-sigma * time_ax)))
        return complex_spec_phased,fid_phased

