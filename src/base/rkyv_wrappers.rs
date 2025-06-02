//! Wrapper that allows changing the generic type of a `PhantomData<NT>`
//!
//! Copied from <https://github.com/rkyv/rkyv_contrib> (MIT-Apache2 licences) which isn’t published yet.

use rkyv::{
    Place,
    rancor::Fallible,
    with::{ArchiveWith, DeserializeWith, SerializeWith},
};
use std::marker::PhantomData;

/// A wrapper that allows for changing the generic type of a `PhantomData<OrigT>` to a `PhantomData<NewT>`.
/// Useful for deriving Archive on abstract types that store generic type information in `PhantomData`s instead of some more fundamental type such as `Vec<T>`.
pub struct CustomPhantom<NT: ?Sized> {
    _data: PhantomData<*const NT>,
}

impl<OrigT: ?Sized, NewT: ?Sized> ArchiveWith<PhantomData<OrigT>> for CustomPhantom<NewT> {
    type Archived = PhantomData<NewT>;
    type Resolver = ();

    #[inline]
    fn resolve_with(_: &PhantomData<OrigT>, _: Self::Resolver, _: Place<Self::Archived>) {}
}

impl<OrigT: ?Sized, NewT: ?Sized, S: Fallible + ?Sized> SerializeWith<PhantomData<OrigT>, S>
    for CustomPhantom<NewT>
{
    #[inline]
    fn serialize_with(_: &PhantomData<OrigT>, _: &mut S) -> Result<Self::Resolver, S::Error> {
        Ok(())
    }
}

impl<OrigT: ?Sized, NewT: ?Sized, D: Fallible + ?Sized>
    DeserializeWith<PhantomData<NewT>, PhantomData<OrigT>, D> for CustomPhantom<NewT>
{
    #[inline]
    fn deserialize_with(_: &PhantomData<NewT>, _: &mut D) -> Result<PhantomData<OrigT>, D::Error> {
        Ok(PhantomData)
    }
}
