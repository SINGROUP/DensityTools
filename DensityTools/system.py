import numpy as np
from ase.io import write
from ase.io.cube import read_cube_data
import ase
from .utils.convolve import gauss, shift


class System(ase.Atoms):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_cube(self, filename, index=-1):
        """
        Save data as cube
        :param filename str: name of file
        :param index int: index of box, default is -1
        """
        if 'box' not in self.info:
            raise ValueError('No box data attached')
        write(filename, self, format='cube', data=self.info['box'][index])

    @classmethod
    def from_cube(cls, filename):
        """
        Returns System object with box data in system.info['box'], read
        from cube file
        :param filename str: name of the file
        :return: instance of the :class:system
        """
        data, atoms = read_cube_data(filename)
        atoms = cls(atoms)
        atoms.info['box'] = data[None, :, :, :]
        return atoms

    @classmethod
    def from_trajectory(cls,
                        dump,
                        rest_atoms_names,
                        box_atoms_names=None,
                        voxl_size=0.2,
                        sigma=None,
                        n_frames=None,
                        density=True):
        """
        Get system from dump trajectory
        :param dump: :class:`ase.Atoms` object in a list or trajectory
        :param rest_atoms_names: list of atom names not in density
        :param box_atoms_names: list of atom names in density,
         defaults to all the atom types.
        :param voxl_size: int, in angstrom, defaults to 0.5
        :param sigma: sigma for gaussian smearing, can be list for each atom
         type in box_atoms_names, defaults to None.
        :param n_frames: last n frames to use, defaults to None.
        :param density: If the value should be divided by volume
        :return: :class:`DensityTools.System` object with atoms defined in
         rest_atoms_names, and a 'box' in :method:`ase.Atoms.info`.
        """
        if n_frames is None:
            n_frames = len(dump)
        elif n_frames > len(dump):
            n_frames = len(dump)
        names = dump[0].arrays['names']
        if box_atoms_names is None:
            box_atoms_names = np.unique(names).tolist()
        n_voxl = np.asarray(np.round(np.linalg.norm(dump[0].get_cell(),
                                                    axis=1) / voxl_size),
                            dtype=int)
        voxl = dump[0].get_cell() / n_voxl
        if density:
            # If density is requested, then it is divided by voxl vol
            voxl_vol = np.dot(voxl[0, :], np.cross(voxl[1, :], voxl[2, :]))
        else:
            voxl_vol = 1
        box = np.zeros([len(box_atoms_names)] + n_voxl.tolist())

        rest_indx = np.where(np.sum([names == x
                                     for x in rest_atoms_names], axis=0))[0]
        rest_atoms = dump[0][rest_indx]

        positions = []
        for count, atom in enumerate(dump[-n_frames:]):
            # print('\r {:.2%}'.format(count/(n_frames-1)), end='')
            if count == 0:
                rest_atoms.positions = atom.positions[rest_indx]
                pos_old = rest_atoms.get_scaled_positions()
                positions = pos_old / n_frames
                # Addition  of stack array to scaled positions finds all
                # scaled coords in all neighbouring cells.
                stack = np.vstack([(np.zeros_like(positions) - 1)[None, :],
                                   (np.zeros_like(positions))[None, :],
                                   (np.zeros_like(positions) + 1)[None, :]])
            else:
                rest_atoms.positions = atom.positions[rest_indx]
                pos_new = rest_atoms.get_scaled_positions()
                pos_new += np.argmin(np.abs(stack + pos_new - pos_old),
                                     axis=0) - 1
                positions += pos_new / n_frames
                pos_old = pos_new
            for indx in range(len(atom)):
                if names[indx] not in box_atoms_names:
                    continue
                com = atom.get_positions()[indx]
                type_ind = box_atoms_names.index(names[indx])
                voxl_indx = np.asarray(np.floor(np.linalg.solve(voxl.T, com)),
                                       dtype=int)
                try:
                    box[type_ind,
                        voxl_indx[0],
                        voxl_indx[1],
                        voxl_indx[2]] += 1 / voxl_vol / n_frames  # / n_water
                except IndexError:
                    voxl_indx[voxl_indx == n_voxl] = 0
                    if np.all(voxl_indx < n_voxl):
                        box[type_ind,
                            voxl_indx[0],
                            voxl_indx[1],
                            voxl_indx[2]] += 1 / voxl_vol / n_frames

        if sigma is not None:
            sigma = np.array(sigma)
            if sigma.shape == ():
                sigma = np.array(sigma[None].tolist() * box.shape[0])
            elif sigma.shape[0] != box.shape[0]:
                raise RuntimeError("Number of sigma values do"
                                   " not match types in the system")
            for i in range(box.shape[0]):
                box[i] = gauss(box[i], sigma=sigma[i], voxl_size=voxl_size)

        rest_atoms = cls(rest_atoms)
        rest_atoms.info['box'] = box
        rest_atoms.info['box_labels'] = box_atoms_names
        return rest_atoms

    @classmethod
    def from_database(cls, fdb, id_):
        """
        retrieve data from a database
        :param fdb: instance of ase database
        :param id_: id of the atoms
        """
        row = fdb.get(id_)
        atoms = cls(row.toatoms())
        if 'box' in row.data.keys():
            n_voxl = np.array(row.data['box_data']['N_voxl'])
            box = row.data['box']
            voxl = atoms.cell.array / n_voxl
            base = int(row.data['box_data']['base'] / voxl[2, 2])
            box_labels = row.data['box_data']['box_atoms_names']

            data = np.zeros([box.shape[0]] + n_voxl.tolist())
            data[:, :, :, base:base+box.shape[3]] = box
            atoms.info["box"] = data
            atoms.info["box_labels"] = box_labels
            atoms.info["base"] = row.data['box_data']['base']
        return atoms

    def fill_with(self, atoms, voxl_size=3, skin=1):
        '''
        Adds atoms box to self
        :param atoms: ase.Atoms with periodic boundary conditions
        :param voxl_size: voxl size in angstrom. Default is 3 angstroms
        :param skin: skin to remove atoms on edge. Default is 1 angstrom
        '''
        cell = self.get_cell()
        if cell.rank != 3:
            raise RuntimeError(f'{self} does not have a valid cell')
        cell = (cell.T * (1 - skin / np.linalg.norm(cell, axis=1))).T
        n_voxl = np.asarray(np.round(np.linalg.norm(cell, axis=1)
                                     / voxl_size), dtype=int)
        voxl = (cell.T / n_voxl).T
        box = np.ones(n_voxl, dtype=bool)

        for i in range(len(self)):
            pos = self.get_positions()[i]
            voxl_indx = np.asarray(np.floor(np.linalg.solve(voxl.T, pos)),
                                   dtype=int)
            voxl_indx[voxl_indx == n_voxl] = 0
            box[voxl_indx[0], voxl_indx[1], voxl_indx[2]] = False
            for ind_i in [-1, 0, 1]:
                for ind_j in [-1, 0, 1]:
                    for ind_k in [-1, 0, 1]:
                        box[(ind_i + voxl_indx[0]) % n_voxl[0],
                            (ind_j + voxl_indx[1]) % n_voxl[1],
                            (ind_k + voxl_indx[2]) % n_voxl[2]] = False

        # checking the number of periodic cells of atoms needed
        indx = np.asarray(np.ceil(np.linalg.solve(atoms.cell.T,
                                                  np.sum(self.cell,
                                                         axis=0))),
                          dtype=int)
        if not np.all(indx == 1):
            atoms *= indx

        # removing molecules which are close to atoms in self
        # or outside the cell
        mol_id_del = set()
        for i, pos in enumerate(atoms.get_positions()):
            voxl_indx = np.asarray(np.floor(np.linalg.solve(voxl.T, pos)),
                                   dtype=int)
            if np.any(voxl_indx >= n_voxl) or np.any(voxl_indx < 0):
                # atoms outside box
                mol_id_del |= set([atoms.get_tags()[i]])
            elif not box[voxl_indx[0], voxl_indx[1], voxl_indx[2]]:
                mol_id_del |= set([atoms.get_tags()[i]])

        # getting indices of all the atoms not in mol_id_del
        ind = []
        for i in atoms.get_tags():
            if i not in mol_id_del:
                ind += np.where(atoms.get_tags() == i)[0].tolist()

        self += atoms[ind]

    def box_to_fractions(self,
                         width,
                         height,
                         base,
                         overlap):
        """Cuts box data into overlapping smaller boxes for training
        """
        if 'box' not in self.info:
            raise ValueError('No box data attached')
        boxes = self.info['box']
        n_voxl = boxes.shape[1:]
        voxl = self.cell.array / n_voxl
        # Size of domain and overlap in the units of number of voxls
        domain_size = np.asarray(np.round([width, width, height]
                                          / np.diag(voxl)),
                                 dtype=int)
        overlap_size = np.asarray(np.round(overlap
                                           / np.linalg.norm(voxl, axis=1)),
                                  dtype=int)[:2]
        overlap_count = np.asarray(np.maximum((np.asarray(n_voxl)
                                               - domain_size / 2)[:2]
                                              // overlap_size,
                                              np.ones(2)),
                                   dtype=int)

        # periodic images of box and rest_cells needed
        indx = np.ceil(((overlap_count - 1)
                        * overlap_size
                        + domain_size[:2])
                       / n_voxl[:2])
        indx = np.asarray(indx.tolist() + [1], dtype=int)

        # if periodic images needed, make them
        if np.any(indx != 1):
            boxes_ = np.zeros(np.array(boxes.shape) * ([1] + indx.tolist()))
            for ind in range(boxes.shape[0]):
                for i in range(indx[0]):
                    for j in range(indx[1]):
                        boxes_[ind, i * n_voxl[0]:(i + 1) * n_voxl[0],
                               j * n_voxl[1]:n_voxl[1] * (1 + j),
                               :] = boxes[ind].copy()
            boxes = boxes_.copy()
        num_instances = int(np.prod(overlap_count))
        density_data = np.zeros([num_instances, boxes.shape[0]]
                                + domain_size.tolist())

        z_min = int(base / voxl[2, 2])
        for ind in range(boxes.shape[0]):
            for i in range(overlap_count[0]):
                for j in range(overlap_count[1]):
                    _ = i * overlap_count[1] + j
                    x_min = i * overlap_size[0]
                    y_min = j * overlap_size[1]
                    x_max = domain_size[0] + i * overlap_size[0]
                    y_max = domain_size[1] + j * overlap_size[1]
                    z_max = z_min + domain_size[2]
                    density_data[_, ind, :, :, :] = boxes[ind,
                                                          x_min:x_max,
                                                          y_min:y_max,
                                                          z_min:z_max]
        return density_data

    def predict(self, func, index=slice(None),
                voxl_size=0.2, sigma=None, skin=0):
        """
        Applies a prediction func to the system.
        :param func: the prediction function
        :param index: indices of the box data to be fed into the prediction
        function
        :param voxl_size: voxl size in angstrom
        :param sigma: sigma for gaussian smearing, can be float, or list of
        floats for each index in the box data
        :param skin: the skin around the prediction to ignore. The predicted
        density is stitched so that the skin is removed, and only the inner
        box covers the final density, placed adjacently.
        :return: predicted density of dim 4, with the same number of indices
        as output of func, and rest three dims are same as the box dims.
        """
        box = self.info['box'][index]
        if sigma is not None:
            sigma = np.array(sigma)
            if sigma.shape == ():
                sigma = np.array(sigma[None].tolist() * box.shape[0])
            elif sigma.shape[0] != box.shape[0]:
                raise RuntimeError("Number of sigma values do"
                                   " not match types in the system")
            for i in range(box.shape[0]):
                box[i] = gauss(box[i], sigma=sigma[i], voxl_size=voxl_size)

        n_voxl = np.array(box.shape[-3:])
        skin_ind = int(skin / voxl_size)

        if skin_ind:
            atom_density = np.zeros([box.shape[0]]
                                    + (np.array(box.shape[1:]) +
                                       2 * skin_ind).tolist())
            box_ = shift(box, [0]+(n_voxl - skin_ind).tolist())

            atom_density[:,
                         :n_voxl[0],
                         :n_voxl[1],
                         :n_voxl[2]] = box_
            atom_density[:,
                         -2*skin_ind:, :, :] = atom_density[:, :2*skin_ind, :, :]
            atom_density[:,
                         :, -2*skin_ind:, :] = atom_density[:, :, :2*skin_ind, :]
            atom_density[:,
                         :, :, -2*skin_ind:] = atom_density[:, :, :, :2*skin_ind]

            out = func(atom_density)[:,
                                     skin_ind:-skin_ind,
                                     skin_ind:-skin_ind,
                                     skin_ind:-skin_ind]
        else:
            out = func(box)

        return out

    def predict_raster(self, func, width, height, base=None, index=slice(None),
                       voxl_size=0.2, sigma=None, skin=0):
        """
        Applies a prediction func to the system, within windows of given
        dimensions.
        :param func: the prediction function
        :param width: width in angstrom of the window
        :param height: height in angstrom of the window
        :param base: the base height in angstrom, from where the prediction
        starts
        :param index: indices of the box data to be fed into the prediction
        function
        :param voxl_size: voxl size in angstrom
        :param sigma: sigma for gaussian smearing, can be float, or list of
        floats for each index in the box data
        :param skin: the skin around the prediction to ignore. The predicted
        density is stitched so that the skin is removed, and only the inner
        box covers the final density, placed adjacently.
        :return: predicted density of dim 4, with the same number of indices
        as output of func, and rest three dims are same as the box dims.
        """
        if "box" not in self.info:
            self = self.__class__.from_trajectory([self])

        if base is None:
            if "base" in self.info.keys():
                base = self.info["base"]
            else:
                raise ValueError(f"Base not found in "
                                 f"{self.__class__.__name__}, provide with"
                                 f"func")

        base_index = int(base / voxl_size)
        box = self.info['box'][index]
        if sigma is not None:
            sigma = np.array(sigma)
            if sigma.shape == ():
                sigma = np.array(sigma[None].tolist() * box.shape[0])
            elif sigma.shape[0] != box.shape[0]:
                raise RuntimeError("Number of sigma values do"
                                   " not match types in the system")
            for i in range(box.shape[0]):
                box[i] = gauss(box[i], sigma=sigma[i], voxl_size=voxl_size)

        # setting up input
        voxl = np.diag(np.array([voxl_size for _ in range(3)]))

        skip = width - (2 * skin)
        # Size of domain and skip in the units of number of voxls
        domain_size = np.asarray(np.round(np.array([width, width, height])
                                          / voxl_size),
                                 dtype=int)
        start_shape = box.shape[-3:]
        if np.any(np.array(box.shape[-3:]) < domain_size):
            print('Warning: box shape smaller than ML input size',
                  'Prediction inaccuracies might be high')
            box_ = np.zeros([box.shape[0]]
                            + np.maximum(domain_size, start_shape).tolist())
            for i in range(box.shape[0]):
                box_[i,
                     :start_shape[0],
                     :start_shape[1],
                     :start_shape[2]] = box[i].copy()
            box = box_.copy()
        n_voxl = np.array(box.shape[-3:])
        skip_size = np.asarray(np.round(skip
                                        / np.linalg.norm(voxl, axis=1)),
                               dtype=int)[:2]
        internal_size = np.asarray(np.round((np.array([skip, skip, height]))
                                            / voxl_size),
                                   dtype=int)
        skip_count = np.asarray(np.maximum((np.asarray(n_voxl)
                                            + domain_size / 2)[:2]
                                           // skip_size,
                                           np.ones(2)),
                                dtype=int)
        edge_size = int(skin / voxl_size)
        # num_instances = int(np.prod(skip_count))

        indx = np.ceil(((skip_count - 1)
                        * skip_size
                        + internal_size[:2])
                       / n_voxl[:2])
        indx = np.asarray(indx.tolist() + [1], dtype=int)

        # if periodic images needed, make them
        if np.any(indx != 1):
            box_ = np.zeros(np.array(box.shape) * ([1] + indx.tolist()))
            for ind in range(box.shape[0]):
                for i in range(indx[0]):
                    for j in range(indx[1]):
                        box_[ind, i * n_voxl[0]:(i + 1) * n_voxl[0],
                             j * n_voxl[1]:n_voxl[1] * (1 + j),
                             :] = box[ind].copy()
            box = box_.copy()

        for i in range(skip_count[0]):
            for j in range(skip_count[1]):
                print(domain_size, base_index, box.shape)
                atom_density = np.zeros([box.shape[0]] + domain_size.tolist())
                sas = (np.array([i * skip_size[0], j * skip_size[1], 0])
                       - [edge_size, edge_size, 0])
                for ind in range(box.shape[0]):
                    hold = shift(box[ind], sas)[:domain_size[0],
                                                :domain_size[1],
                                                base_index:(domain_size[2]
                                                            + base_index)]
                    atom_density[ind, :, :, :] = hold
                hold = func(atom_density)
                # initialise output matrix with same channels
                # as output of network
                if i == 0 and j == 0:
                    density_data = np.zeros([hold.shape[0]] + n_voxl.tolist())
                vec = [internal_size[0] + i * skip_size[0],
                       internal_size[1] + j * skip_size[1]]
                max_x, max_y = np.minimum(vec, [n_voxl[0], n_voxl[1]])
                vec = np.asarray([n_voxl[0] + edge_size - i * skip_size[0],
                                  n_voxl[1] + edge_size - j * skip_size[1]],
                                 dtype=int)
                hold_x, hold_y = np.minimum(vec,
                                            [edge_size + internal_size[0],
                                             edge_size + internal_size[1]])
                _ = hold[:,
                         edge_size:hold_x,
                         edge_size:hold_y,
                         :]
                density_data[:,
                             i * skip_size[0]:max_x,
                             j * skip_size[1]:max_y,
                             base_index:internal_size[2] + base_index] = _
        return density_data[:,
                            :start_shape[0],
                            :start_shape[1],
                            :start_shape[2]]
